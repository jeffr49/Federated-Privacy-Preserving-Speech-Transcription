"""
Run:  cd server && FL_SERVER_AUTOSTART=0 pytest -q
No Hugging Face account or GPU needed -- uses the LocalStore backend.
"""
import json
import os

os.environ["FL_SERVER_AUTOSTART"] = "0"

import numpy as np
import pytest
from fastapi.testclient import TestClient

import aggregator as agg
from app import Config, create_app
from store import LocalStore

KEYS = {"court-a": "ka", "court-b": "kb", "court-c": "kc"}
SHAPES = {"lora_A.weight": (4, 8), "lora_B.weight": (8, 4)}


def make_state(rng, scale=0.1):
    return {k: (rng.standard_normal(s) * scale).astype(np.float32) for k, s in SHAPES.items()}


@pytest.fixture()
def env(tmp_path):
    rng = np.random.default_rng(0)
    store = LocalStore(str(tmp_path / "store"))
    base = make_state(rng)
    store.bootstrap({
        "adapter_model.safetensors": agg.dump_state_bytes(base),
        "adapter_config.json": b'{"r": 4}',
        "version.json": json.dumps({"version": 0}).encode(),
    })
    cfg = Config(store_backend="local", local_store_dir=str(tmp_path / "store"),
                 client_api_keys=KEYS, admin_api_key="admin", min_clients=3, clip_norm=1.0)
    client = TestClient(create_app(cfg, store))
    return client, store, base, rng, cfg


def submit(client, key, delta, base_version=0, n=10):
    return client.post("/submit_delta", headers={"X-API-Key": key},
                       data={"meta": json.dumps({"base_version": base_version, "num_samples": n})},
                       files={"delta": ("d.safetensors", agg.dump_state_bytes(delta), "application/octet-stream")})


def test_auth_required(env):
    client, *_ = env
    assert client.get("/latest").status_code == 401
    assert client.get("/latest", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/latest", headers={"X-API-Key": "ka"}).status_code == 200
    assert client.post("/admin/aggregate", headers={"X-API-Key": "ka"}).status_code == 401


def test_adapter_download_whitelist(env):
    client, *_ = env
    h = {"X-API-Key": "ka"}
    assert client.get("/adapter/adapter_model.safetensors", headers=h).status_code == 200
    assert client.get("/adapter/../../etc/passwd", headers=h).status_code in (404, 422)
    assert client.get("/adapter/secrets.txt", headers=h).status_code == 404


def test_rejects_wrong_schema_and_nan(env):
    client, _, base, rng, _ = env
    bad_keys = {"nope": np.zeros((2, 2), np.float32)}
    assert submit(client, "ka", bad_keys).status_code == 422
    bad_shape = {k: np.zeros((1, 1), np.float32) for k in base}
    assert submit(client, "ka", bad_shape).status_code == 422
    nan = {k: np.full_like(v, np.nan) for k, v in base.items()}
    assert submit(client, "ka", nan).status_code == 422
    assert client.get("/status").json()["pending"] == 0


def test_rejects_stale_base_version(env):
    client, _, base, rng, _ = env
    r = submit(client, "ka", make_state(rng, 0.01), base_version=7)
    assert r.status_code == 409
    assert r.json()["detail"]["current_version"] == 0


def test_round_aggregates_at_min_clients_and_deletes_submissions(env):
    client, store, base, rng, cfg = env
    deltas = [make_state(rng, 0.01) for _ in range(3)]
    counts = [10, 30, 60]
    for (key, d, n) in zip(("ka", "kb", "kc"), deltas, counts):
        r = submit(client, key, d, n=n)
        assert r.status_code == 200
    # background task ran after the 3rd request returned (TestClient runs them synchronously)
    cand = store.root / "candidate"
    assert (cand / "adapter_model.safetensors").exists()
    meta = json.loads((cand / "version.json").read_text())
    assert meta["version"] == 1 and meta["base_version"] == 0 and meta["n_clients"] == 3

    new_state = agg.load_state_bytes((cand / "adapter_model.safetensors").read_bytes())
    w = np.array(counts, dtype=np.float64) / sum(counts)       # all norms are tiny (< clip), so no clipping
    for k in base:
        expected = base[k] + sum(wi * d[k] for wi, d in zip(w, deltas))
        np.testing.assert_allclose(new_state[k], expected, rtol=1e-5, atol=1e-6)

    assert store.pending_clients(0) == []                       # raw deltas deleted after aggregation
    assert store.read_version()["version"] == 0                 # main is NOT promoted by the server; CI does that


def test_server_side_clipping_bounds_a_malicious_client(env):
    client, store, base, rng, cfg = env
    honest = [make_state(rng, 0.001) for _ in range(2)]
    evil = {k: np.full(s, 1e6, np.float32) for k, s in SHAPES.items()}   # enormous update
    submit(client, "ka", honest[0]); submit(client, "kb", honest[1])
    submit(client, "kc", evil, n=10_000)                                  # also tries to claim a huge weight
    new_state = agg.load_state_bytes((store.root / "candidate" / "adapter_model.safetensors").read_bytes())
    moved = agg.l2_norm({k: new_state[k] - base[k] for k in base})
    assert moved <= cfg_clip(env) + 1e-4, f"aggregate moved {moved}, expected <= clip norm"


def cfg_clip(env):
    return env[4].clip_norm


def test_admin_force_aggregate_with_one_client(env):
    client, store, base, rng, _ = env
    submit(client, "ka", make_state(rng, 0.01))
    assert client.get("/status").json()["pending"] == 1
    r = client.post("/admin/aggregate?force=true", headers={"X-API-Key": "admin"})
    assert r.json()["status"] == "aggregated"


def test_resubmission_replaces_not_duplicates(env):
    client, store, base, rng, _ = env
    submit(client, "ka", make_state(rng, 0.01))
    submit(client, "ka", make_state(rng, 0.01))
    assert client.get("/status").json()["pending"] == 1


def test_noise_changes_aggregate_but_stays_finite():
    rng = np.random.default_rng(1)
    ds = [make_state(rng, 0.01) for _ in range(3)]
    clean, _ = agg.aggregate(ds, [1, 1, 1], clip_norm=1.0, noise_mult=0.0)
    noisy, info = agg.aggregate(ds, [1, 1, 1], clip_norm=1.0, noise_mult=0.5, rng=np.random.default_rng(2))
    assert info["noise_std_per_coord"] > 0
    assert any(not np.allclose(clean[k], noisy[k]) for k in clean)
    assert all(np.isfinite(v).all() for v in noisy.values())
