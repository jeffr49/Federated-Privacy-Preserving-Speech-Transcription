"""
Federated aggregation server (FastAPI).

It never sees audio or transcripts -- only clipped LoRA-adapter deltas.

Endpoints
---------
  GET  /                 small HTML status dashboard (public, no client identities)
  GET  /healthz          liveness
  GET  /status           public JSON: current version, pending count, last aggregation
  GET  /latest           (client key) current promoted version metadata
  GET  /adapter/{name}   (client key) download adapter_model.safetensors / adapter_config.json
  POST /submit_delta     (client key) multipart: `meta` (JSON string) + `delta` (safetensors file)
  POST /admin/aggregate  (admin key) force an aggregation round now (?force=true to ignore MIN_CLIENTS)

Round lifecycle
---------------
  clients submit deltas based on version N  ->  when >= MIN_CLIENTS have submitted, the server
  aggregates them into a CANDIDATE adapter (version N+1) on the `candidate` branch  ->  fires a
  GitHub `repository_dispatch` -> the CI workflow evaluates candidate vs. current on a held-out
  set and promotes it to `main` only if WER has not regressed.
"""
from __future__ import annotations

import hmac
import json
import logging
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, Optional

import numpy as np
import requests
from fastapi import BackgroundTasks, Depends, FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response

import aggregator as agg
from store import ADAPTER_FILES, make_store

log = logging.getLogger("fl-server")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@dataclass
class Config:
    store_backend: str = "hub"                 # "hub" | "local"
    local_store_dir: str = "./local_store"
    hf_token: str = ""
    adapter_repo: str = ""
    submissions_repo: str = ""
    client_api_keys: Dict[str, str] = field(default_factory=dict)   # {client_id: api_key}
    admin_api_key: str = ""
    min_clients: int = 3
    max_upload_mb: int = 200
    max_weight_samples: int = 200
    clip_norm: float = 1.0
    noise_mult: float = 0.0
    github_repo: str = ""                      # "owner/repo"
    github_dispatch_token: str = ""

    @classmethod
    def from_env(cls) -> "Config":
        e = os.environ.get
        return cls(
            store_backend=e("STORE_BACKEND", "hub"),
            local_store_dir=e("LOCAL_STORE_DIR", "./local_store"),
            hf_token=e("HF_TOKEN", ""),
            adapter_repo=e("ADAPTER_REPO", ""),
            submissions_repo=e("SUBMISSIONS_REPO", ""),
            client_api_keys=json.loads(e("CLIENT_API_KEYS", "{}") or "{}"),
            admin_api_key=e("ADMIN_API_KEY", ""),
            min_clients=int(e("MIN_CLIENTS", "3")),
            max_upload_mb=int(e("MAX_UPLOAD_MB", "200")),
            max_weight_samples=int(e("MAX_WEIGHT_SAMPLES", "200")),
            clip_norm=float(e("DP_CLIP_NORM", "1.0")),
            noise_mult=float(e("SERVER_NOISE_MULT", "0.0")),
            github_repo=e("GITHUB_REPO", ""),
            github_dispatch_token=e("GITHUB_DISPATCH_TOKEN", ""),
        )


def _safe_eq(a: str, b: str) -> bool:
    return hmac.compare_digest(a.encode("utf-8"), b.encode("utf-8"))


DASHBOARD_HTML = """<!doctype html><html><head><meta charset="utf-8"><title>Court ASR aggregator</title>
<style>body{font-family:system-ui,sans-serif;max-width:640px;margin:48px auto;color:#1A1A2E}
h1{color:#1E2761}.card{background:#EEF3FC;border-radius:10px;padding:16px 20px;margin:12px 0}
.big{font-size:32px;font-weight:700;color:#1E2761}small{color:#5B6178}</style></head><body>
<h1>&#9878; Federated Court ASR &mdash; aggregator</h1>
<div class="card"><small>Promoted adapter version</small><div class="big" id="v">&hellip;</div></div>
<div class="card"><small>Court updates waiting for next round</small><div class="big" id="p">&hellip;</div>
<small id="m"></small></div>
<div class="card"><small>Last aggregation</small><div id="l">&mdash;</div></div>
<small>Only clipped adapter deltas reach this server &mdash; never audio or transcripts.</small>
<script>async function t(){try{const s=await (await fetch('status')).json();
document.getElementById('v').textContent='v'+s.version;
document.getElementById('p').textContent=s.pending;
document.getElementById('m').textContent='round triggers at '+s.min_clients+' courts';
document.getElementById('l').textContent=s.last_aggregation?JSON.stringify(s.last_aggregation):'none yet';}catch(e){}}
t();setInterval(t,5000);</script></body></html>"""


def create_app(cfg: Optional[Config] = None, store=None) -> FastAPI:
    cfg = cfg or Config.from_env()
    store = store or make_store(cfg)
    app = FastAPI(title="Federated Court ASR aggregator")

    agg_lock = threading.Lock()
    state: dict = {"last_aggregation": None, "ref_version": None, "ref_state": None}

    # ------------------------------------------------------------------ auth
    def auth_client(x_api_key: str = Header(default="")) -> str:
        for cid, key in cfg.client_api_keys.items():
            if key and _safe_eq(key, x_api_key):
                return cid
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")

    def auth_admin(x_api_key: str = Header(default="")) -> None:
        if not cfg.admin_api_key or not _safe_eq(cfg.admin_api_key, x_api_key):
            raise HTTPException(status_code=401, detail="admin key required")

    # ------------------------------------------------------------------ helpers
    def reference_state(version: int) -> agg.State:
        """Schema reference for validating deltas (cached per version)."""
        if state["ref_version"] != version:
            state["ref_state"] = agg.load_state_bytes(store.read_adapter_file("adapter_model.safetensors"))
            state["ref_version"] = version
        return state["ref_state"]

    def dispatch_ci(revision: str, new_version: int, base_version: int) -> None:
        if not (cfg.github_repo and cfg.github_dispatch_token):
            log.warning("CI dispatch not configured; run the 'promote-adapter' workflow manually with "
                        "revision=%s (candidate v%d).", revision, new_version)
            return
        try:
            r = requests.post(
                f"https://api.github.com/repos/{cfg.github_repo}/dispatches",
                headers={"Authorization": f"Bearer {cfg.github_dispatch_token}",
                         "Accept": "application/vnd.github+json"},
                json={"event_type": "candidate-adapter-ready",
                      "client_payload": {"revision": revision, "version": new_version, "base_version": base_version}},
                timeout=20,
            )
            log.info("CI dispatch -> HTTP %s", r.status_code)
        except Exception as e:  # never let a CI hiccup break aggregation
            log.error("CI dispatch failed: %s", e)

    def maybe_aggregate(force: bool = False) -> dict:
        if not agg_lock.acquire(blocking=False):
            return {"status": "busy"}
        try:
            cur = store.read_version()
            n = int(cur["version"])
            clients = store.pending_clients(n)
            needed = 1 if force else cfg.min_clients
            if len(clients) < needed:
                return {"status": "waiting", "pending": len(clients), "needed": needed}

            base = agg.load_state_bytes(store.read_adapter_file("adapter_model.safetensors"))
            deltas, counts, used, bad = [], [], [], []
            for cid in clients:
                raw, meta = store.get_submission(n, cid)
                try:
                    d = agg.load_state_bytes(raw)
                    agg.validate_schema(d, base)
                except agg.DeltaRejected as e:
                    log.warning("dropping submission from %s: %s", cid, e)
                    bad.append(cid)
                    continue
                deltas.append(d)
                counts.append(int(meta.get("num_samples", 1)))
                used.append(cid)

            if len(deltas) < needed:
                store.delete_submissions(n, bad)
                return {"status": "waiting", "pending": len(deltas), "needed": needed, "dropped": len(bad)}

            avg, info = agg.aggregate(deltas, counts, cfg.clip_norm, cfg.noise_mult, cfg.max_weight_samples)
            new_state = agg.apply_update(base, avg)
            new_version = n + 1
            version_meta = {
                "version": new_version, "base_version": n, "n_clients": len(used),
                "total_samples": int(sum(counts)),
                "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "clip_norm": cfg.clip_norm, "noise_mult": cfg.noise_mult,
                "status": "candidate",
            }
            files = {
                "adapter_model.safetensors": agg.dump_state_bytes(new_state),
                "adapter_config.json": store.read_adapter_file("adapter_config.json"),
                "version.json": json.dumps(version_meta, indent=2).encode(),
            }
            revision = store.publish_candidate(files)
            store.delete_submissions(n, clients)       # raw client deltas are not retained
            result = {"status": "aggregated", "candidate_version": new_version, "revision": revision,
                      "clients": len(used), "dropped": len(bad), **{k: info[k] for k in ("clipped_clients", "aggregate_norm")}}
            state["last_aggregation"] = {"candidate_version": new_version, "clients": len(used),
                                         "at": version_meta["created_at"]}
            log.info("aggregated %d clients -> candidate v%d @ %s", len(used), new_version, revision)
            dispatch_ci(revision, new_version, n)
            return result
        except Exception as e:
            log.exception("aggregation failed")
            return {"status": "error", "detail": str(e)}
        finally:
            agg_lock.release()

    # ------------------------------------------------------------------ routes
    @app.get("/", response_class=HTMLResponse)
    def dashboard():
        return DASHBOARD_HTML

    @app.get("/healthz")
    def healthz():
        return {"ok": True}

    @app.get("/status")
    def status():
        cur = store.read_version()
        return {"version": cur["version"], "pending": len(store.pending_clients(int(cur["version"]))),
                "min_clients": cfg.min_clients, "last_aggregation": state["last_aggregation"],
                "promoted_meta": {k: cur.get(k) for k in ("created_at", "n_clients", "eval_wer") if k in cur}}

    @app.get("/latest")
    def latest(client_id: str = Depends(auth_client)):
        cur = store.read_version()
        return {**cur, "files": list(ADAPTER_FILES), "clip_norm": cfg.clip_norm}

    @app.get("/adapter/{name}")
    def adapter(name: str, client_id: str = Depends(auth_client)):
        if name not in ADAPTER_FILES:
            raise HTTPException(status_code=404, detail="unknown file")
        return Response(content=store.read_adapter_file(name), media_type="application/octet-stream")

    @app.post("/submit_delta")
    async def submit_delta(background: BackgroundTasks, meta: str = Form(...), delta: UploadFile = File(...),
                           client_id: str = Depends(auth_client)):
        max_bytes = cfg.max_upload_mb * 1024 * 1024
        raw = await delta.read(max_bytes + 1)
        if len(raw) > max_bytes:
            raise HTTPException(status_code=413, detail=f"delta larger than {cfg.max_upload_mb} MB")
        try:
            meta_d = json.loads(meta)
            base_version = int(meta_d["base_version"])
            num_samples = max(1, min(int(meta_d.get("num_samples", 1)), 100_000))
        except Exception:
            raise HTTPException(status_code=422, detail="meta must be JSON with integer base_version (and num_samples)")

        cur = store.read_version()
        if base_version != int(cur["version"]):
            raise HTTPException(status_code=409, detail={
                "message": "stale base_version; pull the latest adapter before submitting",
                "current_version": cur["version"]})
        try:
            d = agg.load_state_bytes(raw)
            agg.validate_schema(d, reference_state(int(cur["version"])))
        except agg.DeltaRejected as e:
            raise HTTPException(status_code=422, detail=str(e))

        store.put_submission(base_version, client_id, raw,
                             {"num_samples": num_samples,
                              "received_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        pending = len(store.pending_clients(base_version))
        background.add_task(maybe_aggregate)
        return {"accepted": True, "client_id": client_id, "version": base_version,
                "pending": pending, "min_clients": cfg.min_clients,
                "norm_before_server_clip": round(agg.l2_norm(d), 6)}

    @app.post("/admin/aggregate")
    def admin_aggregate(force: bool = False, _: None = Depends(auth_admin)):
        return maybe_aggregate(force=force)

    return app


# `uvicorn app:app` entrypoint. Config comes from environment variables (see README).
app = create_app() if os.environ.get("FL_SERVER_AUTOSTART", "1") == "1" else None
