#!/usr/bin/env python3
"""
CI gate: evaluate the CANDIDATE adapter against the CURRENT (promoted) adapter on a fixed held-out
set, and promote the candidate to `main` only if WER has not regressed beyond a tolerance.

Why this exists: federated aggregation can produce a worse model (a noisy round, one bad client,
too-aggressive learning rate). Nothing a court device downloads should ever be an un-evaluated
adapter. The server only writes to the `candidate` branch; THIS script is the only thing that
writes to `main`.

Env / args
----------
  HF_TOKEN               token with write access to ADAPTER_REPO and read access to EVAL_REPO
  ADAPTER_REPO           private model repo holding the adapter (main = promoted, candidate branch)
  EVAL_REPO              private dataset repo with manifest.csv (audio_path,text) + audio files
  BASE_MODEL             default openai/whisper-small
  EVAL_MAX_UTTERANCES    default 100 (CPU runners are slow; 100 clips ~ 5-10 min per adapter)
  PROMOTE_TOLERANCE_WER  default 0.5  (absolute WER points a candidate may be worse by and still pass)
  --revision             exact candidate commit SHA (from the dispatch payload); defaults to branch name
  --dry-run              evaluate and decide, but do not write anything to the Hub

Decision logic is a pure function (`decide`) and is unit-tested in ci/test_ci.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import unicodedata
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Tuple


# ----------------------------------------------------------------------------- pure helpers
def normalize_text(s: str) -> str:
    """NFC-normalise, drop punctuation (Unicode category P*), collapse whitespace.
    NOTE: we deliberately do NOT use regex \\w here -- it does not match Tamil vowel signs
    (combining marks), so stripping [^\\w\\s] would corrupt Tamil text."""
    s = unicodedata.normalize("NFC", str(s))
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith("P"))
    return " ".join(s.lower().split())


def wer_pct(refs: List[str], hyps: List[str]) -> float:
    import jiwer

    pairs = [(normalize_text(r), normalize_text(h)) for r, h in zip(refs, hyps)]
    pairs = [(r, h) for r, h in pairs if r]               # jiwer errors on empty references
    if not pairs:
        raise ValueError("no non-empty reference transcripts in eval set")
    return 100.0 * jiwer.wer([r for r, _ in pairs], [h for _, h in pairs])


def decide(candidate_wer: float, current_wer: float, tolerance: float,
           require_improvement: bool = False) -> Tuple[bool, str]:
    if require_improvement:
        ok = candidate_wer < current_wer
        return ok, (f"candidate {candidate_wer:.2f} {'<' if ok else '>='} current {current_wer:.2f} "
                    f"(strict improvement required)")
    ok = candidate_wer <= current_wer + tolerance
    return ok, (f"candidate {candidate_wer:.2f} vs current {current_wer:.2f} "
                f"(tolerance +{tolerance:.2f}) -> {'PASS' if ok else 'REGRESSION'}")


def eval_set_id(manifest_bytes: bytes, max_n: int) -> str:
    return hashlib.sha256(manifest_bytes + str(max_n).encode()).hexdigest()[:16]


# ----------------------------------------------------------------------------- model eval
def evaluate_adapter(adapter_dir: Path, base_model: str, eval_root: Path, rows, batch_size: int = 8) -> float:
    import librosa
    import torch
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(base_model, language="ta", task="transcribe")
    base = WhisperForConditionalGeneration.from_pretrained(base_model)      # fresh base per adapter (PEFT mutates it)
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()

    refs, hyps = [], []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        audios = [librosa.load(str(eval_root / r["audio_path"]), sr=16000, mono=True)[0] for r in chunk]
        feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt").input_features
        with torch.no_grad():
            out = model.generate(input_features=feats, language="ta", task="transcribe",
                                 max_new_tokens=128, num_beams=1)
        hyps += processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        refs += [r["text"] for r in chunk]
        print(f"  evaluated {min(i + batch_size, len(rows))}/{len(rows)}", flush=True)
    return wer_pct(refs, hyps)


# ----------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--revision", default="", help="candidate commit SHA (default: 'candidate' branch)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    import pandas as pd
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    token = os.environ["HF_TOKEN"]
    adapter_repo = os.environ["ADAPTER_REPO"]
    eval_repo = os.environ["EVAL_REPO"]
    base_model = os.environ.get("BASE_MODEL", "openai/whisper-small")
    max_n = int(os.environ.get("EVAL_MAX_UTTERANCES", "100"))
    tol = float(os.environ.get("PROMOTE_TOLERANCE_WER", "0.5"))
    strict = os.environ.get("REQUIRE_IMPROVEMENT", "0") == "1"
    cand_rev = a.revision or "candidate"
    api = HfApi(token=token)

    def read_json(rev: str, name: str = "version.json") -> dict:
        p = hf_hub_download(adapter_repo, name, revision=rev, token=token, repo_type="model")
        return json.loads(Path(p).read_text("utf-8"))

    main_meta = read_json("main")
    cand_meta = read_json(cand_rev)
    cur_v, cand_v = int(main_meta["version"]), int(cand_meta["version"])
    print(f"main = v{cur_v}, candidate = v{cand_v} (base v{cand_meta.get('base_version')}) @ {cand_rev}")

    if int(cand_meta.get("base_version", -1)) != cur_v or cand_v != cur_v + 1:
        print("STALE candidate: main has moved since this candidate was aggregated. Skipping (no changes made).")
        return 0

    patterns = ["adapter_model.safetensors", "adapter_config.json", "version.json"]
    cand_dir = Path(snapshot_download(adapter_repo, revision=cand_rev, token=token, allow_patterns=patterns))
    cur_dir = Path(snapshot_download(adapter_repo, revision="main", token=token, allow_patterns=patterns))

    eval_root = Path(snapshot_download(eval_repo, repo_type="dataset", token=token))
    manifest_bytes = (eval_root / "manifest.csv").read_bytes()
    df = pd.read_csv(eval_root / "manifest.csv").head(max_n)
    rows = df.to_dict("records")
    set_id = eval_set_id(manifest_bytes, max_n)
    print(f"eval set: {len(rows)} utterances (id {set_id})")

    print("evaluating CANDIDATE ...")
    cand_wer = evaluate_adapter(cand_dir, base_model, eval_root, rows)

    if main_meta.get("eval_set_id") == set_id and main_meta.get("eval_wer") is not None:
        cur_wer = float(main_meta["eval_wer"])
        print(f"reusing cached WER for current v{cur_v}: {cur_wer:.2f}")
    else:
        print("evaluating CURRENT ...")
        cur_wer = evaluate_adapter(cur_dir, base_model, eval_root, rows)

    ok, reason = decide(cand_wer, cur_wer, tol, strict)
    print(reason)

    summary = (f"### Adapter promotion gate\n\n| | WER % |\n|---|---|\n| current v{cur_v} | {cur_wer:.2f} |\n"
               f"| candidate v{cand_v} | {cand_wer:.2f} |\n\n**{'PROMOTED' if ok else 'REJECTED'}** — {reason}\n")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        Path(os.environ["GITHUB_STEP_SUMMARY"]).write_text(summary, encoding="utf-8")

    if a.dry_run:
        print("--dry-run: nothing written to the Hub.")
        return 0

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if ok:
        # re-check for a race right before writing
        if int(read_json("main")["version"]) != cur_v:
            print("main moved during evaluation; aborting promotion.")
            return 0
        meta = {**cand_meta, "status": "promoted", "promoted_at": now, "eval_wer": round(cand_wer, 4),
                "eval_wer_previous": round(cur_wer, 4), "eval_set_id": set_id, "eval_n": len(rows)}
        out = Path("promote_out")
        out.mkdir(exist_ok=True)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            (out / name).write_bytes((cand_dir / name).read_bytes())
        (out / "version.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        api.upload_folder(folder_path=str(out), repo_id=adapter_repo, repo_type="model", revision="main",
                          commit_message=f"promote v{cand_v}: WER {cur_wer:.2f} -> {cand_wer:.2f}")
        api.create_tag(repo_id=adapter_repo, tag=f"v{cand_v}", revision="main", repo_type="model", exist_ok=True)
        print(f"PROMOTED v{cand_v} to main and tagged v{cand_v}.")
    else:
        rec = {"version": cand_v, "rejected_at": now, "candidate_wer": round(cand_wer, 4),
               "current_wer": round(cur_wer, 4), "reason": reason, "candidate_revision": cand_rev}
        api.upload_file(path_or_fileobj=json.dumps(rec, indent=2).encode(), path_in_repo=f"rejected/v{cand_v}.json",
                        repo_id=adapter_repo, repo_type="model", revision="main",
                        commit_message=f"record rejection of candidate v{cand_v}")
        print(f"REJECTED v{cand_v}; main stays at v{cur_v}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
