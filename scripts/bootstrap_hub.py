#!/usr/bin/env python3
"""
One-time setup: create the three private Hub repos and publish your existing fine-tuned adapter
as version 0 (the starting point every court downloads).

  export HF_TOKEN=hf_xxx            # a token with WRITE access (https://huggingface.co/settings/tokens)
  python scripts/bootstrap_hub.py \
      --adapter-dir ./runs/whisper-small_lora_r32/final_adapter \
      --adapter-repo  <user>/court-asr-adapters \
      --submissions-repo <user>/court-asr-submissions \
      --eval-repo <user>/court-asr-eval

Safe to re-run: repos are created with exist_ok, but version 0 is only published if the adapter repo
has no version.json yet (it will never silently overwrite a promoted model).
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi
from huggingface_hub.utils import EntryNotFoundError
from safetensors.numpy import load_file


try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", required=True)
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--submissions-repo", required=True)
    ap.add_argument("--eval-repo", required=True)
    ap.add_argument("--token", default=None, help="Hugging Face write token (defaults to HF_TOKEN from .env or env)")
    a = ap.parse_args()

    token = a.token or os.environ.get("HF_TOKEN")
    if not token:
        sys.exit("set HF_TOKEN first (either in .env, export HF_TOKEN=..., or pass --token hf_...)")
    api = HfApi(token=token)
    d = Path(a.adapter_dir)

    for f in ("adapter_model.safetensors", "adapter_config.json"):
        if not (d / f).exists():
            sys.exit(f"{d / f} not found -- point --adapter-dir at the folder saved by train_whisper_lora_local.py")

    # The server does its maths in float32 numpy; refuse adapters it cannot read (e.g. bf16).
    try:
        tensors = load_file(str(d / "adapter_model.safetensors"))
    except Exception as e:
        sys.exit(f"adapter is not readable as float tensors by the server ({e}).\n"
                 f"Re-save it in float32:  python -c \"import torch; from safetensors.torch import load_file, save_file; "
                 f"s=load_file('{d}/adapter_model.safetensors'); save_file({{k:v.float().contiguous() for k,v in s.items()}}, "
                 f"'{d}/adapter_model.safetensors')\"")
    n_params = sum(int(np.prod(v.shape)) for v in tensors.values())
    print(f"adapter OK: {len(tensors)} tensors, {n_params / 1e6:.2f} M parameters, "
          f"{(d / 'adapter_model.safetensors').stat().st_size / 1e6:.1f} MB")

    api.create_repo(a.adapter_repo, repo_type="model", private=True, exist_ok=True)
    api.create_repo(a.submissions_repo, repo_type="dataset", private=True, exist_ok=True)
    api.create_repo(a.eval_repo, repo_type="dataset", private=True, exist_ok=True)
    print("repos ready (all private)")

    try:
        api.hf_hub_download(a.adapter_repo, "version.json", repo_type="model", revision="main")
        print("version.json already exists on main -- NOT overwriting. Nothing more to do.")
        return
    except EntryNotFoundError:
        pass
    except Exception as e:
        # a brand-new empty repo raises a "not found"-style error; anything else we surface
        print(f"(no existing version.json: {type(e).__name__})")

    api.upload_file(path_or_fileobj=str(d / "adapter_model.safetensors"), path_in_repo="adapter_model.safetensors",
                    repo_id=a.adapter_repo, repo_type="model", commit_message="v0: initial fine-tuned adapter")
    api.upload_file(path_or_fileobj=str(d / "adapter_config.json"), path_in_repo="adapter_config.json",
                    repo_id=a.adapter_repo, repo_type="model", commit_message="v0: adapter config")
    api.upload_file(path_or_fileobj=json.dumps({"version": 0, "status": "promoted", "note": "initial adapter"}).encode(),
                    path_in_repo="version.json", repo_id=a.adapter_repo, repo_type="model",
                    commit_message="v0: version metadata")
    api.create_tag(repo_id=a.adapter_repo, tag="v0", repo_type="model", exist_ok=True)
    print("published v0. Next: build + upload the eval set (ci/make_eval_set.py), then deploy the Space.")


if __name__ == "__main__":
    main()
