"""
fl_client_sync.py -- the ONLY code in the client that talks to the aggregation server.

What crosses the network:
  - DOWN: the promoted LoRA adapter (adapter_model.safetensors + adapter_config.json)
  - UP:   one clipped LoRA-delta file + a tiny JSON (base_version, num_samples)
Audio and transcripts are never passed to any function in this module.

Configuration (env vars or constructor args):
  FL_SERVER_URL   e.g. https://<user>-court-asr-aggregator.hf.space
  FL_CLIENT_ID    e.g. court-chennai   (informational; the server identifies you by API key)
  FL_API_KEY      the key the server admin issued to this court
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import requests
from safetensors.numpy import save as st_save

ADAPTER_FILES = ("adapter_model.safetensors", "adapter_config.json")


def _to_np(x) -> np.ndarray:
    """Accept torch tensors or numpy arrays; always return contiguous float32 numpy."""
    if hasattr(x, "detach"):
        x = x.detach().cpu().float().numpy()
    return np.ascontiguousarray(np.asarray(x, dtype=np.float32))


def serialize_delta(delta: Dict[str, object]) -> bytes:
    return st_save({k: _to_np(v) for k, v in delta.items()})


class StaleBaseError(RuntimeError):
    """Server has a newer promoted adapter than the one this delta was computed against."""

    def __init__(self, current_version: int):
        super().__init__(f"server is at v{current_version}; pull the latest adapter first")
        self.current_version = current_version


class FLClient:
    def __init__(self, server_url: Optional[str] = None, client_id: Optional[str] = None,
                 api_key: Optional[str] = None, cache_dir: str = "./fl_cache", timeout: int = 120):
        self.url = (server_url or os.environ.get("FL_SERVER_URL", "")).rstrip("/")
        self.client_id = client_id or os.environ.get("FL_CLIENT_ID", "unnamed-court")
        self.api_key = api_key or os.environ.get("FL_API_KEY", "")
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout
        if not self.url:
            raise ValueError("FL server URL not set (FL_SERVER_URL).")

    @property
    def _h(self) -> dict:
        return {"X-API-Key": self.api_key}

    # ---------------------------------------------------------------- pull
    def latest(self) -> dict:
        r = requests.get(f"{self.url}/latest", headers=self._h, timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def pull(self, force: bool = False) -> tuple[Path, int]:
        """Download the promoted adapter if we don't already have that version.
        Returns (adapter_dir, version). adapter_dir is loadable with PeftModel.from_pretrained()."""
        info = self.latest()
        version = int(info["version"])
        dest = self.cache / f"adapter_v{version}"
        if force or not all((dest / f).exists() for f in ADAPTER_FILES):
            dest.mkdir(parents=True, exist_ok=True)
            for name in ADAPTER_FILES:
                r = requests.get(f"{self.url}/adapter/{name}", headers=self._h, timeout=self.timeout)
                r.raise_for_status()
                (dest / name).write_bytes(r.content)
        return dest, version

    # ---------------------------------------------------------------- push
    def push_delta(self, delta: Dict[str, object], base_version: int, num_samples: int) -> dict:
        """Upload a (already clipped) LoRA delta computed against `base_version`."""
        payload = serialize_delta(delta)
        r = requests.post(
            f"{self.url}/submit_delta", headers=self._h, timeout=self.timeout,
            data={"meta": json.dumps({"base_version": int(base_version), "num_samples": int(num_samples),
                                      "client_id": self.client_id})},
            files={"delta": ("delta.safetensors", payload, "application/octet-stream")},
        )
        if r.status_code == 409:
            detail = r.json().get("detail", {})
            raise StaleBaseError(int(detail.get("current_version", -1)))
        if r.status_code >= 400:
            raise RuntimeError(f"server rejected update ({r.status_code}): {r.text[:300]}")
        return r.json()

    def status(self) -> dict:
        r = requests.get(f"{self.url}/status", timeout=self.timeout)
        r.raise_for_status()
        return r.json()
