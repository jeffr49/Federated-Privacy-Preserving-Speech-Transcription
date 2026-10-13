#!/usr/bin/env python3
"""
Build the fixed held-out evaluation set used by the CI promotion gate.

Takes the same manifest.parquet your training scripts use, picks N utterances from the *valid*
split (never train), spread round-robin across districts so no single accent dominates, copies
the audio, and writes manifest.csv (audio_path,text,district).

  python ci/make_eval_set.py --data-root D:/tamil_fl_data --out ./eval_set --n 100

Then upload it as a PRIVATE dataset repo (this is the repo named in EVAL_REPO):

  huggingface-cli upload <your-user>/court-asr-eval ./eval_set . --repo-type dataset --private

Keep this set fixed. If you change it, the CI cache of the current model's WER is invalidated
automatically (the set id is a hash of manifest.csv + EVAL_MAX_UTTERANCES) and both adapters are re-scored.
Use only data you are licensed to redistribute to your own private repo (AI4Bharat releases can be
research-only) -- never real court recordings.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--out", default="./eval_set")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--max-seconds", type=float, default=30.0, help="Whisper's window is 30 s")
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()

    root = Path(a.data_root)
    m = pd.read_parquet(root / "manifest.parquet")
    label = m["normalized"].where(m["normalized"].astype(str).str.strip() != "", m["text"])
    m = m.assign(label_text=label)
    m = m[(m["split"] == "valid") & (m["label_text"].astype(str).str.strip() != "") & (m["duration"] <= a.max_seconds)]
    if m.empty:
        raise SystemExit("no usable validation utterances found")

    m = m.sample(frac=1.0, random_state=a.seed)
    per_district = {d: g.reset_index(drop=True) for d, g in m.groupby("district", dropna=False)}
    picked, i = [], 0
    while len(picked) < min(a.n, len(m)) and per_district:
        for d in list(per_district):
            g = per_district[d]
            if i < len(g):
                picked.append(g.iloc[i])
                if len(picked) >= a.n:
                    break
            else:
                per_district.pop(d)
        i += 1

    out = Path(a.out)
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rows = []
    for j, r in enumerate(picked):
        src = root / r["audio_path"]
        dst_rel = f"audio/{j:04d}{src.suffix}"
        shutil.copy2(src, out / dst_rel)
        rows.append({"audio_path": dst_rel, "text": r["label_text"], "district": r["district"]})
    pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False, encoding="utf-8")
    print(f"wrote {len(rows)} utterances from {len({r['district'] for r in rows})} districts -> {out}")


if __name__ == "__main__":
    main()
