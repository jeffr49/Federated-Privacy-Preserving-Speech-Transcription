#!/usr/bin/env python3
"""Development helper: stand-in for the CI gate when running with STORE_BACKEND=local.
Copies store/candidate/* over store/main/ (i.e. 'promotes' the candidate). NO evaluation is done --
in production only ci/evaluate_and_promote.py may write to main."""
import json
import shutil
import sys
from pathlib import Path

store = Path(sys.argv[1] if len(sys.argv) > 1 else "./local_store")
cand, main = store / "candidate", store / "main"
meta = json.loads((cand / "version.json").read_text())
meta["status"] = "promoted"
for f in ("adapter_model.safetensors", "adapter_config.json"):
    shutil.copy2(cand / f, main / f)
(main / "version.json").write_text(json.dumps(meta, indent=2))
print(f"promoted candidate -> main (now v{meta['version']})")
