#!/usr/bin/env python3
"""Development helper: create a LOCAL store (no Hugging Face) holding your adapter as version 0.

  python scripts/bootstrap_local.py --adapter-dir ./runs/.../final_adapter --store ./local_store
  # then run the server with STORE_BACKEND=local LOCAL_STORE_DIR=./local_store
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
from store import LocalStore  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--adapter-dir", required=True)
ap.add_argument("--store", default="./local_store")
a = ap.parse_args()
d = Path(a.adapter_dir)
LocalStore(a.store).bootstrap({
    "adapter_model.safetensors": (d / "adapter_model.safetensors").read_bytes(),
    "adapter_config.json": (d / "adapter_config.json").read_bytes(),
    "version.json": json.dumps({"version": 0, "status": "promoted"}).encode(),
})
print(f"local store ready at {a.store} (v0)")
