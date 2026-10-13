#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_whisper_lora_local.py
===========================
Whisper + LoRA fine-tuning for Tamil ASR (IndicVoices manifest), rewritten from the
Kaggle notebook `train-tamil-indic.ipynb` for an UNATTENDED local run.

What it does, in order (everything lands in  <out-root>/<run-name>/ ):
  1. Pre-flight checks (GPU, dataset path, audio files, disk) -> fails in seconds, not hours.
  2. Trains the LoRA adapter (same data pipeline / LoRA config / LR as the notebook).
       - auto-resumes from the newest checkpoint if the run is restarted
       - auto-recovers from CUDA out-of-memory (enables grad-checkpointing / halves the
         per-device batch and raises grad-accumulation so the effective batch stays the same)
       - saves an adapter every epoch  -> adapters/epoch_XX/
       - saves the best adapter        -> final_adapter/   (same name as the notebook)
       - writes the full training log  -> train_log_history.csv / train_log.jsonl
  3. Evaluates the adapter on the FULL valid + test splits and writes
       WER, CER, substitution/deletion/insertion rates, per-district and per-duration
       breakdowns, per-utterance predictions, zero-shot-vs-finetuned comparison, speed (RTFx).
  4. Saves all the graphs to plots/  (re-generate any time with --plots-only, no GPU needed).

Quick start
-----------
  pip install torch --index-url https://download.pytorch.org/whl/cu124   # CUDA build, see pytorch.org
  pip install transformers peft accelerate librosa soundfile pandas pyarrow matplotlib rapidfuzz

  # 1) 5-minute benchmark: finds a batch size that fits, prints time/epoch and total-time estimates
  python train_whisper_lora_local.py --data-root D:/tamil_fl_data --benchmark

  # 2) the real, unattended run (re-run the SAME command to resume after a crash/reboot)
  python train_whisper_lora_local.py --data-root D:/tamil_fl_data --epochs 10

  # bigger model
  python train_whisper_lora_local.py --data-root D:/tamil_fl_data --model openai/whisper-medium --epochs 4

  # only evaluate an existing adapter / only redraw graphs
  python train_whisper_lora_local.py --data-root D:/tamil_fl_data --eval-only
  python train_whisper_lora_local.py --plots-only --run-name whisper-small_lora_r32
"""
from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import shutil
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, Subset

import librosa
import soundfile as sf
from rapidfuzz.distance import Levenshtein
from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, get_peft_model

# ----------------------------------------------------------------------------------------
# Constants (same as the notebook)
# ----------------------------------------------------------------------------------------
LANGUAGE = "ta"
LANGUAGE_NAME = "Tamil"
TASK = "transcribe"
TARGET_SR = 16000
MAX_LABEL_LEN = 448  # Whisper decoder hard limit
DEFAULT_DATA_ROOT = os.environ.get("TAMIL_DATA_ROOT", "./tamil_fl_data")
BENCH_STEPS = 40

plt.rcParams.update({
    "figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.3,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
})


# ----------------------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------------------
def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


class Tee:
    """Mirror stdout/stderr into run.log so nothing is lost while you are away."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            try:
                st.write(s)
                st.flush()
            except Exception:
                pass
        return len(s)

    def flush(self):
        for st in self.streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self.streams[0], name)


def keep_awake() -> None:
    """Stop Windows from idle-sleeping while the script runs (also set 'lid close = do nothing')."""
    if platform.system() == "Windows":
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
            ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
            log("Windows keep-awake enabled.")
        except Exception as e:  # pragma: no cover
            log(f"Could not enable keep-awake: {e}")
    else:
        log("Not Windows: prevent sleep yourself, e.g.  systemd-inhibit --what=sleep python ...")


def cleanup_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def is_oom(e: BaseException) -> bool:
    if isinstance(e, torch.cuda.OutOfMemoryError):
        return True
    return isinstance(e, RuntimeError) and "out of memory" in str(e).lower()


def jsonable(o: Any):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return None if np.isnan(o) else float(o)
    if isinstance(o, (np.ndarray,)):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, float) and math.isnan(o):
        return None
    return str(o)


def dump_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=jsonable), encoding="utf-8")


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
@dataclass
class Cfg:
    data_root: str
    out_root: str
    run_name: str
    model_id: str
    epochs: int
    batch_size: int
    grad_accum: int
    grad_checkpointing: bool
    lr: float
    warmup_steps: int
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_targets: List[str]
    train_hour_cap: float
    eval_subset: int
    evals_per_epoch: int
    early_stop_patience: int
    workers: int
    seed: int
    baseline_subset: int
    eval_batch_size: int
    gen_max_len: Optional[int]
    max_crash_retries: int
    benchmark: bool
    eval_only: bool
    adapter: Optional[str]
    plots_only: bool
    skip_eval: bool


def parse_args() -> Cfg:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="folder containing manifest.parquet and the audio")
    p.add_argument("--out-root", default="./runs")
    p.add_argument("--run-name", default=None, help="keep it fixed to be able to resume; default derives from model")
    p.add_argument("--model", default="openai/whisper-small")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=None, help="per-device batch (default depends on model)")
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--grad-checkpointing", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", choices=["qv", "all"], default="qv",
                   help="qv = q_proj,v_proj (notebook). all = q,k,v,out_proj,fc1,fc2 (more capacity)")
    p.add_argument("--train-hour-cap", type=float, default=15.0, help="hours per district (notebook: 15)")
    p.add_argument("--eval-subset", type=int, default=600, help="valid rows used for in-training eval")
    p.add_argument("--evals-per-epoch", type=int, default=2)
    p.add_argument("--early-stop-patience", type=int, default=8,
                   help="number of evals without improvement before stopping (0 = off)")
    p.add_argument("--workers", type=int, default=None, help="dataloader workers (default: cpu_count-2, max 5; train + eval loaders each keep this many alive)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-subset", type=int, default=1000,
                   help="valid rows for the zero-shot baseline comparison (0 = skip)")
    p.add_argument("--eval-batch-size", type=int, default=None)
    p.add_argument("--gen-max-len", type=int, default=None,
                   help="override generation length cap (default derived from label-length p99)")
    p.add_argument("--max-crash-retries", type=int, default=3)
    p.add_argument("--benchmark", action="store_true", help=f"run {BENCH_STEPS} steps, print time estimates, exit")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--adapter", default=None, help="adapter dir for --eval-only (default: run_dir/final_adapter)")
    p.add_argument("--plots-only", action="store_true")
    p.add_argument("--skip-eval", action="store_true", help="train only, skip the full evaluation")
    a = p.parse_args()

    big = any(k in a.model.lower() for k in ("medium", "large"))
    bs = a.batch_size or (4 if big else 8)
    accum = a.grad_accum or (4 if big else 2)
    gc_flag = a.grad_checkpointing if a.grad_checkpointing is not None else big
    run_name = a.run_name or f"{Path(a.model).name}_lora_r{a.lora_r}"
    targets = ["q_proj", "v_proj"] if a.lora_targets == "qv" else \
        ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]
    workers = a.workers if a.workers is not None else max(2, min(5, (os.cpu_count() or 4) - 2))
    return Cfg(
        data_root=a.data_root, out_root=a.out_root, run_name=run_name, model_id=a.model, epochs=a.epochs,
        batch_size=bs, grad_accum=accum, grad_checkpointing=gc_flag, lr=a.lr, warmup_steps=a.warmup_steps,
        lora_r=a.lora_r, lora_alpha=a.lora_alpha, lora_dropout=a.lora_dropout, lora_targets=targets,
        train_hour_cap=a.train_hour_cap, eval_subset=a.eval_subset, evals_per_epoch=a.evals_per_epoch,
        early_stop_patience=a.early_stop_patience, workers=workers, seed=a.seed,
        baseline_subset=a.baseline_subset, eval_batch_size=a.eval_batch_size or (8 if big else 16),
        gen_max_len=a.gen_max_len, max_crash_retries=a.max_crash_retries, benchmark=a.benchmark,
        eval_only=a.eval_only, adapter=a.adapter, plots_only=a.plots_only, skip_eval=a.skip_eval,
    )


# ----------------------------------------------------------------------------------------
# Metrics (rapidfuzz Levenshtein == what jiwer uses internally, so numbers match the notebook)
# ----------------------------------------------------------------------------------------
def edit_counts(ref_tokens, hyp_tokens):
    """returns (substitutions, deletions, insertions) between two token sequences"""
    s = d = i = 0
    for op in Levenshtein.editops(ref_tokens, hyp_tokens):
        if op.tag == "replace":
            s += 1
        elif op.tag == "delete":
            d += 1
        else:
            i += 1
    return s, d, i


def utterance_stats(ref: str, hyp: str) -> Dict[str, int]:
    rw, hw = ref.split(), hyp.split()
    ws, wd, wi = edit_counts(rw, hw)
    rc, hc = ref.strip(), hyp.strip()
    cs, cd, ci = edit_counts(rc, hc)
    return {"w_sub": ws, "w_del": wd, "w_ins": wi, "w_n": len(rw), "hyp_words": len(hw),
            "c_sub": cs, "c_del": cd, "c_ins": ci, "c_n": len(rc)}


def corpus_wer_cer(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    st = [utterance_stats(r, h) for r, h in zip(refs, hyps) if r.strip()]
    if not st:
        return {"wer": float("nan"), "cer": float("nan")}
    df = pd.DataFrame(st)
    return {"wer": 100.0 * (df.w_sub.sum() + df.w_del.sum() + df.w_ins.sum()) / max(df.w_n.sum(), 1),
            "cer": 100.0 * (df.c_sub.sum() + df.c_del.sum() + df.c_ins.sum()) / max(df.c_n.sum(), 1)}


def add_derived(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["w_err"] = df.w_sub + df.w_del + df.w_ins
    df["c_err"] = df.c_sub + df.c_del + df.c_ins
    df["utt_wer"] = df.w_err / df.w_n.clip(lower=1)
    df["utt_cer"] = df.c_err / df.c_n.clip(lower=1)
    return df


def agg_metrics(g: pd.DataFrame) -> pd.Series:
    wn, cn = max(g.w_n.sum(), 1), max(g.c_n.sum(), 1)
    return pd.Series({
        "n_utts": len(g),
        "hours": g.duration.sum() / 3600.0,
        "WER": 100.0 * g.w_err.sum() / wn,
        "CER": 100.0 * g.c_err.sum() / cn,
        "sub_pct": 100.0 * g.w_sub.sum() / wn,
        "del_pct": 100.0 * g.w_del.sum() / wn,
        "ins_pct": 100.0 * g.w_ins.sum() / wn,
        "exact_match_pct": 100.0 * (g.w_err == 0).mean(),
        "median_utt_WER": 100.0 * g.utt_wer.median(),
        "len_ratio_hyp_over_ref": g.hyp_words.sum() / wn,
        "hit_length_cap_pct": 100.0 * g.hit_cap.mean(),
    })


DUR_BINS = [0, 2, 4, 6, 8, 12, 16, 20, 30, np.inf]
DUR_LABELS = ["0-2s", "2-4s", "4-6s", "6-8s", "8-12s", "12-16s", "16-20s", "20-30s", "30s+"]


def build_tables(preds: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
    """preds: split -> DataFrame with the columns written by transcribe_df()"""
    out: Dict[str, Any] = {"overall": {}, "district": {}, "duration": {}}
    for split, df in preds.items():
        df = add_derived(df)
        preds[split] = df
        out["overall"][split] = agg_metrics(df)
        if "district" in df and df["district"].notna().any():
            out["district"][split] = df.groupby("district", observed=True).apply(agg_metrics).sort_values("WER")
        d = df.copy()
        d["dur_bucket"] = pd.cut(d.duration, bins=DUR_BINS, labels=DUR_LABELS, right=False)
        out["duration"][split] = d.groupby("dur_bucket", observed=True).apply(agg_metrics)
    out["overall"] = pd.DataFrame(out["overall"]).T
    return out


def top_word_errors(df: pd.DataFrame, k: int = 60) -> pd.DataFrame:
    subs, dels, inss = Counter(), Counter(), Counter()
    for ref, hyp in zip(df["ref"], df["hyp"]):
        rw, hw = ref.split(), str(hyp).split()
        for op in Levenshtein.editops(rw, hw):
            if op.tag == "replace":
                subs[(rw[op.src_pos], hw[op.dest_pos])] += 1
            elif op.tag == "delete":
                dels[rw[op.src_pos]] += 1
            else:
                inss[hw[op.dest_pos]] += 1
    rows = [("substitution", r, h, c) for (r, h), c in subs.most_common(k)]
    rows += [("deletion", r, "", c) for r, c in dels.most_common(k)]
    rows += [("insertion", "", h, c) for h, c in inss.most_common(k)]
    return pd.DataFrame(rows, columns=["type", "reference_word", "hypothesis_word", "count"])


# ----------------------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------------------
def load_audio(path: str) -> np.ndarray:
    try:
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != TARGET_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    except Exception:
        audio, _ = librosa.load(path, sr=TARGET_SR, mono=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError("empty or non-finite audio")
    return audio


class TamilASRDataset(Dataset):
    """Training / in-training-eval dataset. A corrupt audio file must never kill a multi-day run:
    on failure it logs the path and substitutes a neighbouring sample."""

    def __init__(self, df: pd.DataFrame, processor, bad_log: Optional[str] = None):
        self.paths = df["audio_full_path"].tolist()
        self.texts = df["label_text"].tolist()
        self.fe = processor.feature_extractor
        self.tok = processor.tokenizer
        self.bad_log = bad_log

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        last_err = None
        for attempt in range(6):
            j = (idx + attempt * 7919) % len(self.paths)
            try:
                audio = load_audio(self.paths[j])
                feats = self.fe(audio, sampling_rate=TARGET_SR).input_features[0]
                labels = self.tok(self.texts[j]).input_ids
                return {"input_features": feats, "labels": labels}
            except Exception as e:  # noqa: BLE001
                last_err = e
                if self.bad_log:
                    try:
                        with open(self.bad_log, "a", encoding="utf-8") as f:
                            f.write(f"{self.paths[j]}\t{type(e).__name__}: {e}\n")
                    except Exception:
                        pass
        raise RuntimeError(f"6 consecutive unreadable audio files near index {idx}: {last_err}")


class EvalAudioDataset(Dataset):
    """Inference dataset. Unreadable audio -> silent features + ok=False (row excluded from metrics)."""

    def __init__(self, df: pd.DataFrame, processor):
        self.paths = df["audio_full_path"].tolist()
        self.fe = processor.feature_extractor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        try:
            feats = self.fe(load_audio(self.paths[idx]), sampling_rate=TARGET_SR).input_features[0]
            return {"input_features": feats, "idx": idx, "ok": True}
        except Exception:  # noqa: BLE001
            feats = self.fe(np.zeros(TARGET_SR, dtype=np.float32), sampling_rate=TARGET_SR).input_features[0]
            return {"input_features": feats, "idx": idx, "ok": False}


def eval_collate(batch):
    return {
        "input_features": torch.from_numpy(np.stack([b["input_features"] for b in batch])),
        "idx": [b["idx"] for b in batch],
        "ok": [b["ok"] for b in batch],
    }


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features):
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


@dataclass
class Data:
    processor: Any
    manifest: pd.DataFrame
    train_df: pd.DataFrame
    valid_df: pd.DataFrame
    valid_eval_subset: pd.DataFrame
    label_lens: np.ndarray
    gen_cap: int


def prepare_data(cfg: Cfg, run_dir: Path) -> Data:
    manifest_path = Path(cfg.data_root) / "manifest.parquet"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            f"Download the Kaggle dataset locally and pass --data-root <folder containing manifest.parquet>")
    manifest = pd.read_parquet(manifest_path)
    need = {"audio_path", "split", "district", "duration", "normalized", "text"}
    if need - set(manifest.columns):
        raise KeyError(f"manifest.parquet is missing columns: {sorted(need - set(manifest.columns))}")
    manifest["audio_full_path"] = manifest["audio_path"].apply(lambda p: os.path.join(cfg.data_root, p))
    log(f"Total rows: {len(manifest)}\n{manifest['split'].value_counts().to_string()}")

    # sanity: are the audio files really there?
    probe = manifest.sample(min(200, len(manifest)), random_state=0)
    missing = [p for p in probe["audio_full_path"] if not os.path.exists(p)]
    if len(missing) > 0.02 * len(probe):
        raise FileNotFoundError(f"{len(missing)}/{len(probe)} probed audio files are missing, e.g. {missing[:3]}. "
                                f"Check --data-root.")
    for p in probe["audio_full_path"].head(5):
        load_audio(p)  # raises if the decoder stack is broken
    log("Audio probe OK.")

    # per-district cap on the train split (identical logic to the notebook)
    train_mask = manifest["split"] == "train"
    keep_idx = []
    for _, group in manifest[train_mask].groupby("district", dropna=False):
        group = group.sample(frac=1.0, random_state=42)
        kept = group[group["duration"].cumsum() / 3600 <= cfg.train_hour_cap]
        keep_idx.extend(kept.index.tolist())
    manifest = manifest.loc[keep_idx + manifest[~train_mask].index.tolist()].reset_index(drop=True)
    log(f"After train cap of {cfg.train_hour_cap}h/district:\n{manifest['split'].value_counts().to_string()}")

    processor = WhisperProcessor.from_pretrained(cfg.model_id, language=LANGUAGE_NAME, task=TASK)

    def pick(norm, text):
        if isinstance(norm, str) and norm.strip():
            return norm
        return text if isinstance(text, str) else ""

    manifest["label_text"] = [pick(a, b) for a, b in zip(manifest["normalized"], manifest["text"])]
    n0 = len(manifest)
    manifest = manifest[manifest["label_text"].str.strip() != ""].reset_index(drop=True)
    if len(manifest) != n0:
        log(f"Dropped {n0 - len(manifest)} rows with empty transcripts.")

    log("Measuring label lengths (tokens)...")
    texts = manifest["label_text"].tolist()
    lens = []
    for i in range(0, len(texts), 2000):
        lens.extend(len(x) for x in processor.tokenizer(texts[i:i + 2000]).input_ids)
    manifest["_label_len"] = lens
    n0 = len(manifest)
    manifest = manifest[manifest["_label_len"] <= MAX_LABEL_LEN].reset_index(drop=True)
    log(f"Length filter (<= {MAX_LABEL_LEN} tokens): dropped {n0 - len(manifest)} of {n0} rows.")
    label_lens = manifest["_label_len"].to_numpy()
    p50, p95, p99 = (float(np.percentile(label_lens, q)) for q in (50, 95, 99))
    gen_cap = cfg.gen_max_len or int(min(444, max(128, math.ceil(p99 * 1.1))))
    log(f"Label length in tokens: p50={p50:.0f}  p95={p95:.0f}  p99={p99:.0f}  max={label_lens.max()}  "
        f"-> generation cap = {gen_cap}")
    if p95 > 100:
        log(f"NOTE: the Kaggle notebook used generation_max_length=100, but {100 * (label_lens > 100).mean():.1f}% "
            f"of references are longer than 100 tokens -> its eval WER was inflated by truncated hypotheses.")

    train_df = manifest[manifest["split"] == "train"].reset_index(drop=True)
    valid_df = manifest[manifest["split"] == "valid"].reset_index(drop=True)
    subset = valid_df.sample(n=min(cfg.eval_subset, len(valid_df)), random_state=42).reset_index(drop=True)
    log(f"Train: {len(train_df)} | Valid (full): {len(valid_df)} | in-training eval subset: {len(subset)}")
    return Data(processor, manifest, train_df, valid_df, subset, label_lens, gen_cap)


# ----------------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------------
def use_bf16() -> bool:
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def build_base_model(model_id: str):
    # Kept in fp32. Mixed precision is handled by autocast (TrainingArguments bf16=True at train
    # time, torch.autocast(...) at eval time) - NOT by pre-casting the weights. Pre-casting a Whisper
    # model's conv layers to bf16 triggers a known dtype-mismatch bug inside model.generate() ("Input
    # type (float) and bias type (struct c10::BFloat16) should be the same"), which would otherwise
    # surface the first time the eval/generate path runs - i.e. at the first in-training checkpoint of
    # an unattended run, or during the standalone evaluation pass.
    model = WhisperForConditionalGeneration.from_pretrained(model_id)
    model.generation_config.language = LANGUAGE
    model.generation_config.task = TASK
    model.generation_config.forced_decoder_ids = None
    return model


def build_peft_model(cfg: Cfg):
    model = build_base_model(cfg.model_id)
    lora = LoraConfig(r=cfg.lora_r, lora_alpha=cfg.lora_alpha, target_modules=cfg.lora_targets,
                      lora_dropout=cfg.lora_dropout, bias="none")
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()
    return model


# ----------------------------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------------------------
class ConsoleProgress(TrainerCallback):
    """One readable line per log step with an ETA (tqdm is disabled so run.log stays clean)."""

    def on_train_begin(self, args, state, control, **kw):
        self.t0, self.s0 = time.time(), state.global_step

    def on_log(self, args, state, control, logs=None, **kw):
        if not logs or "loss" not in logs:
            return
        done = max(state.global_step - self.s0, 1)
        elapsed = time.time() - self.t0
        eta = (state.max_steps - state.global_step) * elapsed / done
        print(f"[progress] step {state.global_step}/{state.max_steps} | epoch {(state.epoch or 0):.2f} | "
              f"loss {logs['loss']:.4f} | lr {logs.get('learning_rate', float('nan')):.2e} | "
              f"elapsed {elapsed / 3600:.2f}h | ETA {eta / 3600:.2f}h", flush=True)

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            print(f"[eval] step {state.global_step} epoch {(state.epoch or 0):.2f} | "
                  f"WER {metrics.get('eval_wer', float('nan')):.2f} | CER {metrics.get('eval_cer', float('nan')):.2f} | "
                  f"loss {metrics.get('eval_loss', float('nan')):.4f}", flush=True)


class JsonlLog(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kw):
        if logs:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"step": state.global_step, "epoch": state.epoch, **logs}, default=jsonable) + "\n")


class EpochAdapterSaver(TrainerCallback):
    def __init__(self, root: Path):
        self.root = root

    def on_epoch_end(self, args, state, control, model=None, **kw):
        try:
            path = self.root / f"epoch_{int(round(state.epoch)):02d}"
            model.save_pretrained(path)
            log(f"Saved epoch adapter -> {path}")
        except Exception as e:  # noqa: BLE001
            log(f"Epoch adapter save failed (non-fatal): {e}")


class BenchCallback(TrainerCallback):
    def __init__(self):
        self.t = {}

    def on_step_end(self, args, state, control, **kw):
        if state.global_step in (10, BENCH_STEPS):
            torch.cuda.synchronize()
            self.t[state.global_step] = time.perf_counter()


# ----------------------------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------------------------
def make_ladder(cfg: Cfg):
    """(per-device batch, grad accum, grad checkpointing) settings to try in order on CUDA OOM.
    The effective batch size stays constant."""
    eff = cfg.batch_size * cfg.grad_accum
    bs, gc_on = cfg.batch_size, cfg.grad_checkpointing
    ladder = [(bs, max(1, round(eff / bs)), gc_on)]
    if not gc_on:
        ladder.append((bs, max(1, round(eff / bs)), True))
    while bs > 1:
        bs //= 2
        ladder.append((bs, max(1, round(eff / bs)), True))
    return ladder


def run_training_attempt(cfg, run_dir, data: Data, bs, accum, use_gc, bf16, benchmark):
    torch.cuda.reset_peak_memory_stats()
    proc = data.processor
    model = build_peft_model(cfg)
    bad_log = str(run_dir / "bad_audio.txt")
    train_ds = TamilASRDataset(data.train_df, proc, bad_log)
    eval_ds = TamilASRDataset(data.valid_eval_subset, proc, bad_log)

    eff = bs * accum
    steps_per_epoch = math.ceil(len(train_ds) / eff)
    eval_steps = max(50, steps_per_epoch // cfg.evals_per_epoch)
    ckpt_dir = run_dir / ("_benchmark" if benchmark else "checkpoints")

    def compute_metrics(pred):
        ids = pred.predictions[0] if isinstance(pred.predictions, tuple) else pred.predictions
        lab = pred.label_ids.copy()
        lab[lab == -100] = proc.tokenizer.pad_token_id
        hyp = proc.tokenizer.batch_decode(ids, skip_special_tokens=True)
        ref = proc.tokenizer.batch_decode(lab, skip_special_tokens=True)
        m = corpus_wer_cer(ref, hyp)
        return {"wer": m["wer"], "cer": m["cer"]}

    kw: Dict[str, Any] = dict(
        output_dir=str(ckpt_dir),
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=max(4, min(16, bs * 2)),
        gradient_accumulation_steps=accum,
        learning_rate=cfg.lr,
        warmup_steps=cfg.warmup_steps,
        num_train_epochs=cfg.epochs,
        logging_steps=50,
        fp16=not bf16,
        bf16=bf16,
        predict_with_generate=True,
        generation_max_length=data.gen_cap,
        generation_num_beams=1,
        report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        dataloader_num_workers=cfg.workers,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=cfg.workers > 0,
        dataloader_prefetch_factor=4 if cfg.workers > 0 else None,
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else None,
        disable_tqdm=True,
        seed=cfg.seed,
    )
    callbacks: List[TrainerCallback] = []
    if benchmark:
        kw.update(max_steps=BENCH_STEPS, eval_strategy="no", save_strategy="no", logging_steps=10)
        bench = BenchCallback()
        callbacks.append(bench)
    else:
        kw.update(eval_strategy="steps", eval_steps=eval_steps, save_strategy="steps", save_steps=eval_steps,
                  save_total_limit=3, load_best_model_at_end=True, metric_for_best_model="wer",
                  greater_is_better=False)
        callbacks += [JsonlLog(run_dir / "train_log.jsonl"), EpochAdapterSaver(run_dir / "adapters")]
        if cfg.early_stop_patience > 0:
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=cfg.early_stop_patience,
                                                   early_stopping_threshold=0.05))
    callbacks.append(ConsoleProgress())

    trainer = Seq2SeqTrainer(
        model=model, args=Seq2SeqTrainingArguments(**kw), train_dataset=train_ds, eval_dataset=eval_ds,
        data_collator=DataCollatorSpeechSeq2SeqWithPadding(processor=proc), compute_metrics=compute_metrics,
        processing_class=proc.feature_extractor, callbacks=callbacks,
    )

    log(f"TRAIN attempt: per-device batch={bs} x accum={accum} (effective {eff}), grad-checkpointing={use_gc}, "
        f"{'bf16' if bf16 else 'fp16'}, steps/epoch={steps_per_epoch}, eval/save every {eval_steps} steps")

    resume = None if benchmark else get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if resume:
        log(f"Resuming from {resume}")
    t0 = time.time()
    result = trainer.train(resume_from_checkpoint=resume)
    train_min = (time.time() - t0) / 60
    peak = torch.cuda.max_memory_allocated() / 2 ** 30

    if benchmark:
        sec_step = (bench.t[BENCH_STEPS] - bench.t[10]) / (BENCH_STEPS - 10)
        eval_sec = float("nan")
        try:  # time a small eval so the total-time estimate includes evaluation overhead
            n = min(96, len(eval_ds))
            t1 = time.time()
            trainer.evaluate(eval_dataset=Subset(eval_ds, list(range(n))))
            eval_sec = (time.time() - t1) * len(eval_ds) / n
        except Exception as e:  # noqa: BLE001
            log(f"(benchmark eval timing skipped: {e})")
        print("\n================ BENCHMARK RESULT ================")
        print(f"Config that fits : per-device batch {bs} x accum {accum}, grad-checkpointing={use_gc}")
        print(f"Peak GPU memory  : {peak:.2f} GB")
        print(f"Speed            : {sec_step:.2f} s / optimizer step ({eff / sec_step:.1f} samples/s)")
        print(f"Steps per epoch  : {steps_per_epoch}  ->  {sec_step * steps_per_epoch / 60:.0f} min / epoch (training only)")
        if math.isnan(eval_sec):
            print("In-training eval : timing failed - totals below are TRAINING TIME ONLY (add ~10-20% for evals)")
            eval_sec = 0.0
        else:
            print(f"In-training eval : ~{eval_sec / 60:.1f} min per eval on {len(eval_ds)} rows "
                  f"x {cfg.evals_per_epoch} per epoch")
        for e in (5, 8, 10, 15, 20):
            tot = (sec_step * steps_per_epoch * e + eval_sec * cfg.evals_per_epoch * e) / 3600
            print(f"   {e:>2} epochs  ->  ~{tot:5.1f} h")
        print("(First 10 steps are excluded; real speed can drift lower if the laptop heats up.)")
        print("===================================================\n")
        shutil.rmtree(ckpt_dir, ignore_errors=True)
        return {"benchmark": True}

    # ---- finalize -----------------------------------------------------------------------------------
    final = run_dir / "final_adapter"  # NOTE: best-on-eval-subset weights (load_best_model_at_end=True)
    final.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(final)
    proc.save_pretrained(final)
    log_df = pd.DataFrame(trainer.state.log_history)
    log_df.to_csv(run_dir / "train_log_history.csv", index=False)
    summary = {
        "finished": datetime.now().isoformat(timespec="seconds"),
        "train_minutes_this_session": train_min,
        "global_step": trainer.state.global_step,
        "epochs_completed": trainer.state.epoch,
        "best_eval_wer_subset": trainer.state.best_metric,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_gpu_mem_gb": peak,
        "per_device_batch": bs, "grad_accum": accum, "grad_checkpointing": use_gc,
        "train_metrics": result.metrics,
    }
    dump_json(summary, run_dir / "train_summary.json")
    (run_dir / "TRAINING_DONE").write_text(json.dumps(summary, default=jsonable), encoding="utf-8")
    log(f"Training finished in {train_min:.1f} min. Adapter saved -> {final}")
    del trainer, model
    return summary


def train_with_recovery(cfg, run_dir, data, bf16):
    ladder, i, crashes = make_ladder(cfg), 0, 0
    while i < len(ladder):
        bs, accum, use_gc = ladder[i]
        oom = False
        try:
            return run_training_attempt(cfg, run_dir, data, bs, accum, use_gc, bf16, cfg.benchmark)
        except KeyboardInterrupt:
            raise
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                oom = True
                log(f"CUDA OOM at batch {bs} x accum {accum}, grad-ckpt={use_gc}: {str(e)[:160]}")
            else:
                crashes += 1
                log(f"Training crashed ({crashes}/{cfg.max_crash_retries}): {type(e).__name__}: {e}")
                traceback.print_exc()
                if cfg.benchmark or crashes > cfg.max_crash_retries:
                    raise
        cleanup_cuda()
        if oom:
            i += 1
            if i < len(ladder):
                log("Retrying with a smaller-memory setting (effective batch size unchanged)...")
        else:
            log("Waiting 30 s, then resuming from the last checkpoint...")
            time.sleep(30)
    raise RuntimeError("Out of memory even at per-device batch 1 with gradient checkpointing. "
                       "Use a smaller model (whisper-small) or close other GPU programs.")


# ----------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def transcribe_df(model, processor, df: pd.DataFrame, cfg: Cfg, max_new_tokens: int, name: str,
                  bf16: bool) -> pd.DataFrame:
    # Model stays fp32; autocast does the precision drop for speed, same as during training.
    amp_dtype = torch.bfloat16 if bf16 else torch.float16
    eos = processor.tokenizer.eos_token_id
    loader = DataLoader(EvalAudioDataset(df, processor), batch_size=cfg.eval_batch_size, shuffle=False,
                        num_workers=cfg.workers, collate_fn=eval_collate, pin_memory=True,
                        prefetch_factor=4 if cfg.workers > 0 else None)

    def gen(feats):
        oom = False
        try:
            with torch.autocast("cuda", dtype=amp_dtype):
                out = model.generate(input_features=feats, language=LANGUAGE, task=TASK,
                                     max_new_tokens=max_new_tokens, num_beams=1, do_sample=False)
            return out.cpu()
        except Exception as e:  # noqa: BLE001
            if not (is_oom(e) and feats.shape[0] > 1):
                raise
            oom = True
        if oom:  # split the batch in two and retry
            cleanup_cuda()
            h = feats.shape[0] // 2
            a, b = gen(feats[:h]), gen(feats[h:])
            width = max(a.shape[1], b.shape[1])
            pad = lambda t: torch.nn.functional.pad(t, (0, width - t.shape[1]), value=eos)  # noqa: E731
            return torch.cat([pad(a), pad(b)])

    hyps, hit_cap, ok = [""] * len(df), [False] * len(df), [True] * len(df)
    t0 = time.perf_counter()
    for bi, batch in enumerate(loader):
        feats = batch["input_features"].to("cuda", non_blocking=True)
        out = gen(feats)
        texts = processor.tokenizer.batch_decode(out, skip_special_tokens=True)
        for k, idx in enumerate(batch["idx"]):
            hyps[idx] = texts[k].strip()
            hit_cap[idx] = bool(out[k, -1].item() != eos)
            ok[idx] = batch["ok"][k]
        if bi % 50 == 0:
            done = min((bi + 1) * cfg.eval_batch_size, len(df))
            log(f"  [{name}] {done}/{len(df)} utterances")
    wall = time.perf_counter() - t0

    rows = []
    for r, (ref, hyp) in enumerate(zip(df["label_text"], hyps)):
        rows.append(utterance_stats(ref, hyp))
    res = pd.DataFrame(rows)
    res.insert(0, "split", name)
    res.insert(1, "audio_path", df["audio_path"].values)
    res.insert(2, "district", df["district"].values)
    res.insert(3, "duration", df["duration"].values)
    res.insert(4, "ref", df["label_text"].values)
    res.insert(5, "hyp", hyps)
    res["hit_cap"] = hit_cap
    res["audio_ok"] = ok
    res = res[res["audio_ok"]].reset_index(drop=True)
    res.attrs["wall_sec"] = wall
    res.attrs["audio_sec"] = float(df["duration"].sum())
    return res


def run_evaluation(cfg: Cfg, run_dir: Path, data: Data, adapter_dir: Path, bf16: bool) -> None:
    ev_dir = run_dir / "eval"
    ev_dir.mkdir(exist_ok=True)
    log(f"=== FULL EVALUATION with adapter {adapter_dir} ===")
    base = build_base_model(cfg.model_id).to("cuda")
    model = PeftModel.from_pretrained(base, str(adapter_dir)).eval()
    cleanup_cuda()

    manifest = data.manifest
    preds: Dict[str, pd.DataFrame] = {}
    speed: Dict[str, Any] = {}
    for split in ("valid", "test"):
        df = manifest[manifest["split"] == split].reset_index(drop=True)
        if df.empty:
            continue
        log(f"Transcribing {split}: {len(df)} utterances ({df['duration'].sum() / 3600:.2f} h audio)")
        res = transcribe_df(model, data.processor, df, cfg, data.gen_cap, split, bf16)
        speed[split] = {"wall_min": res.attrs["wall_sec"] / 60, "audio_hours": res.attrs["audio_sec"] / 3600,
                        "RTFx": res.attrs["audio_sec"] / max(res.attrs["wall_sec"], 1e-9)}
        preds[split] = res
        res.to_csv(ev_dir / f"predictions_{split}.csv", index=False, encoding="utf-8-sig")

    baseline_cmp = None
    if cfg.baseline_subset > 0 and "valid" in preds:
        sub = data.valid_df.sample(n=min(cfg.baseline_subset, len(data.valid_df)), random_state=cfg.seed)
        log(f"Zero-shot baseline (adapter disabled) on {len(sub)} valid utterances")
        with model.disable_adapter():
            base_res = transcribe_df(model, data.processor, sub.reset_index(drop=True), cfg, data.gen_cap,
                                     "zero_shot", bf16)
        base_res.to_csv(ev_dir / "predictions_zero_shot_valid_subset.csv", index=False, encoding="utf-8-sig")
        ft = preds["valid"][preds["valid"]["audio_path"].isin(set(base_res["audio_path"]))]
        baseline_cmp = {"zero_shot": base_res, "fine_tuned": ft}

    del model, base
    cleanup_cuda()
    write_reports(cfg, run_dir, preds, baseline_cmp, speed)


# ----------------------------------------------------------------------------------------
# Reports + graphs
# ----------------------------------------------------------------------------------------
def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_label_lengths(lens: np.ndarray, gen_cap: int, path: Path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lens, bins=60, color="#4c78a8")
    for q, c in ((50, "#54a24b"), (95, "#f58518"), (99, "#e45756")):
        v = np.percentile(lens, q)
        ax.axvline(v, ls="--", color=c, label=f"p{q} = {v:.0f}")
    ax.axvline(gen_cap, color="k", label=f"generation cap = {gen_cap}")
    ax.set_xlabel("reference length (Whisper tokens)")
    ax.set_ylabel("utterances")
    ax.set_title("Transcript length distribution (tokens)")
    ax.legend()
    _save(fig, path)


def plot_training(log_csv: Path, out: Path):
    df = pd.read_csv(log_csv)
    if "epoch" not in df:
        return
    tr = df[df["loss"].notna()] if "loss" in df else df.iloc[0:0]
    ev = df[df["eval_loss"].notna()] if "eval_loss" in df else df.iloc[0:0]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    if len(tr):
        ax.plot(tr["epoch"], tr["loss"], alpha=0.25, color="#4c78a8")
        ax.plot(tr["epoch"], tr["loss"].rolling(10, min_periods=1).mean(), color="#4c78a8", label="train loss (smoothed)")
    if len(ev):
        ax.plot(ev["epoch"], ev["eval_loss"], "o-", color="#e45756", label="eval loss (subset)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Train vs eval loss (a widening gap = overfitting)")
    ax.legend()
    _save(fig, out / "01_loss_curves.png")

    if len(ev) and "eval_wer" in ev:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(ev["epoch"], ev["eval_wer"], "o-", color="#4c78a8", label="WER")
        if "eval_cer" in ev:
            ax.plot(ev["epoch"], ev["eval_cer"], "s-", color="#f58518", label="CER")
        b = ev.loc[ev["eval_wer"].idxmin()]
        ax.annotate(f"best WER {b['eval_wer']:.2f}\n(epoch {b['epoch']:.1f})", (b["epoch"], b["eval_wer"]),
                    textcoords="offset points", xytext=(10, 15), arrowprops={"arrowstyle": "->"})
        ax.set_xlabel("epoch")
        ax.set_ylabel("error rate (%)")
        ax.set_title("Validation WER / CER during training (600-row subset)")
        ax.legend()
        _save(fig, out / "02_eval_wer_cer.png")

    if len(tr) and ("learning_rate" in tr or "grad_norm" in tr):
        fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        if "learning_rate" in tr:
            axes[0].plot(tr["epoch"], tr["learning_rate"], color="#54a24b")
        axes[0].set_ylabel("learning rate")
        if "grad_norm" in tr:
            axes[1].plot(tr["epoch"], tr["grad_norm"], color="#b279a2", alpha=0.6)
        axes[1].set_ylabel("grad norm")
        axes[1].set_xlabel("epoch")
        axes[0].set_title("LR schedule and gradient norm")
        _save(fig, out / "03_lr_and_grad_norm.png")


def plot_eval(tables: Dict[str, Any], preds: Dict[str, pd.DataFrame], baseline_cmp, out: Path):
    splits = list(preds.keys())
    colors = {"valid": "#4c78a8", "test": "#f58518"}

    # 10 - per-district WER + CER
    if tables["district"]:
        first = tables["district"][splits[0]]
        order = first.index.tolist()
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
        for ax, metric in zip(axes, ("WER", "CER")):
            w = 0.8 / len(splits)
            for k, s in enumerate(splits):
                t = tables["district"][s].reindex(order)
                ax.bar(np.arange(len(order)) + k * w, t[metric], w, label=s, color=colors.get(s))
            ax.set_xticks(np.arange(len(order)) + 0.4 - w / 2)
            ax.set_xticklabels(order, rotation=45, ha="right")
            ax.set_ylabel(f"{metric} (%)")
            ax.set_title(f"{metric} by district")
            ax.legend()
        _save(fig, out / "10_wer_cer_by_district.png")

    # 11 - error breakdown
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 2.2]})
    ov = tables["overall"]
    bottom = np.zeros(len(ov))
    for col, c, lab in (("sub_pct", "#4c78a8", "substitutions"), ("del_pct", "#e45756", "deletions"),
                        ("ins_pct", "#f58518", "insertions")):
        axes[0].bar(ov.index, ov[col], bottom=bottom, color=c, label=lab)
        bottom += ov[col].to_numpy()
    axes[0].set_ylabel("% of reference words")
    axes[0].set_title("Word-error composition (overall)")
    axes[0].legend()
    if tables["district"]:
        t = tables["district"][splits[0]]
        bottom = np.zeros(len(t))
        for col, c in (("sub_pct", "#4c78a8"), ("del_pct", "#e45756"), ("ins_pct", "#f58518")):
            axes[1].bar(t.index, t[col], bottom=bottom, color=c)
            bottom += t[col].to_numpy()
        axes[1].set_xticks(range(len(t)))
        axes[1].set_xticklabels(t.index, rotation=45, ha="right")
        axes[1].set_title(f"Word-error composition by district ({splits[0]})")
    _save(fig, out / "11_error_breakdown.png")

    # 12 - WER vs duration
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax2 = ax.twinx()
    for s in splits:
        t = tables["duration"][s]
        ax.plot(t.index.astype(str), t["WER"], "o-", label=f"WER {s}", color=colors.get(s))
    t0 = tables["duration"][splits[0]]
    ax2.bar(t0.index.astype(str), t0["n_utts"], alpha=0.15, color="gray")
    ax2.set_ylabel("utterances (bars)")
    ax2.grid(False)
    ax.set_xlabel("utterance duration")
    ax.set_ylabel("WER (%)")
    ax.set_title("WER vs utterance duration")
    ax.legend()
    _save(fig, out / "12_wer_vs_duration.png")

    # 13 - per-utterance WER distribution
    v = preds[splits[0]]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(v["utt_wer"].clip(upper=1.5), bins=45, color="#4c78a8")
    ax.set_xlabel("per-utterance WER (clipped at 1.5)")
    ax.set_ylabel("utterances")
    ax2 = ax.twinx()
    xs = np.sort(v["utt_wer"].clip(upper=1.5).to_numpy())
    ax2.plot(xs, np.arange(1, len(xs) + 1) / len(xs), color="#e45756")
    ax2.set_ylabel("cumulative fraction")
    ax2.grid(False)
    ax.set_title(f"Distribution of utterance-level WER ({splits[0]}) - exact matches: "
                 f"{100 * (v['w_err'] == 0).mean():.1f}%")
    _save(fig, out / "13_utterance_wer_distribution.png")

    # 14 - hypothesis/reference length ratio (hallucination / truncation check)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratio = (v["hyp_words"] / v["w_n"].clip(lower=1)).clip(upper=3)
    ax.hist(ratio, bins=60, color="#54a24b")
    ax.axvline(1.0, color="k", ls="--")
    ax.set_xlabel("hypothesis words / reference words (clipped at 3)")
    ax.set_ylabel("utterances")
    ax.set_title(f"Length ratio ({splits[0]}) - hit generation cap: {100 * v['hit_cap'].mean():.2f}%\n"
                 "<1 = dropped words / truncation, >1 = repetition or hallucination")
    _save(fig, out / "14_length_ratio.png")

    # 15 - zero-shot vs fine-tuned
    if baseline_cmp:
        rows = {k: agg_metrics(add_derived(df)) for k, df in baseline_cmp.items()}
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        names = list(rows.keys())
        for ax, metric in zip(axes, ("WER", "CER")):
            vals = [rows[n][metric] for n in names]
            bars = ax.bar(["zero-shot Whisper", "LoRA fine-tuned"], vals, color=["#bab0ac", "#4c78a8"])
            for b_, val in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, val, f"{val:.1f}", ha="center", va="bottom")
            ax.set_ylabel(f"{metric} (%)")
            ax.set_title(f"{metric} on the same {int(rows[names[0]]['n_utts'])} valid utterances")
        _save(fig, out / "15_zero_shot_vs_finetuned.png")


def write_reports(cfg: Cfg, run_dir: Path, preds: Dict[str, pd.DataFrame], baseline_cmp, speed) -> None:
    ev_dir, plot_dir = run_dir / "eval", run_dir / "plots"
    ev_dir.mkdir(exist_ok=True)
    plot_dir.mkdir(exist_ok=True)
    tables = build_tables(preds)  # also adds derived columns into preds

    tables["overall"].to_csv(ev_dir / "overall_by_split.csv", encoding="utf-8-sig")
    for s, t in tables["district"].items():
        t.to_csv(ev_dir / f"per_district_{s}.csv", encoding="utf-8-sig")
    for s, t in tables["duration"].items():
        t.to_csv(ev_dir / f"per_duration_{s}.csv", encoding="utf-8-sig")
    if "valid" in preds:
        top_word_errors(preds["valid"]).to_csv(ev_dir / "top_word_errors_valid.csv", index=False, encoding="utf-8-sig")
        preds["valid"].query("w_n >= 3").nlargest(150, "utt_wer")[
            ["audio_path", "district", "duration", "ref", "hyp", "utt_wer"]
        ].to_csv(ev_dir / "worst_utterances_valid.csv", index=False, encoding="utf-8-sig")

    summary: Dict[str, Any] = {
        "run": cfg.run_name, "model": cfg.model_id,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "overall_by_split": tables["overall"].to_dict(orient="index"),
        "per_district": {s: t.to_dict(orient="index") for s, t in tables["district"].items()},
        "per_duration": {s: t.astype(float).to_dict(orient="index") for s, t in tables["duration"].items()},
        "inference_speed": speed,
    }
    if baseline_cmp:
        summary["zero_shot_vs_finetuned_same_utterances"] = {
            k: agg_metrics(add_derived(df)).to_dict() for k, df in baseline_cmp.items()}
    if (run_dir / "train_summary.json").exists():
        summary["training"] = json.loads((run_dir / "train_summary.json").read_text(encoding="utf-8"))
    dump_json(summary, ev_dir / "metrics_summary.json")

    log("Drawing graphs...")
    if (run_dir / "train_log_history.csv").exists():
        plot_training(run_dir / "train_log_history.csv", plot_dir)
    if preds:
        plot_eval(tables, preds, baseline_cmp, plot_dir)

    print("\n================ FINAL RESULTS ================")
    print(tables["overall"][["n_utts", "hours", "WER", "CER", "sub_pct", "del_pct", "ins_pct",
                             "exact_match_pct", "hit_length_cap_pct"]].round(2).to_string())
    if baseline_cmp:
        for k, df in baseline_cmp.items():
            m = agg_metrics(add_derived(df))
            print(f"{k:>11}: WER {m['WER']:.2f} | CER {m['CER']:.2f}  (same {int(m['n_utts'])} valid utterances)")
    print(f"Everything is in: {run_dir}")
    print("===============================================\n")


def regenerate_plots(cfg: Cfg, run_dir: Path) -> None:
    ev, plot_dir = run_dir / "eval", run_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    preds: Dict[str, pd.DataFrame] = {}
    for s in ("valid", "test"):
        f = ev / f"predictions_{s}.csv"
        if f.exists():
            preds[s] = pd.read_csv(f, encoding="utf-8-sig", keep_default_na=False, na_values=[])
    if not preds:
        log("No eval/predictions_*.csv found - drawing training graphs only.")
        if (run_dir / "train_log_history.csv").exists():
            plot_training(run_dir / "train_log_history.csv", plot_dir)
        return
    baseline = None
    fz = ev / "predictions_zero_shot_valid_subset.csv"
    if fz.exists() and "valid" in preds:
        z = pd.read_csv(fz, encoding="utf-8-sig", keep_default_na=False, na_values=[])
        z["hit_cap"] = z["hit_cap"].map(lambda x: str(x) == "True")
        baseline = {"zero_shot": z,
                    "fine_tuned": preds["valid"][preds["valid"]["audio_path"].isin(set(z["audio_path"]))]}
    for df in preds.values():
        df["hit_cap"] = df["hit_cap"].map(lambda x: str(x) == "True")
    speed = {}
    old_summary = ev / "metrics_summary.json"
    if old_summary.exists():
        speed = json.loads(old_summary.read_text(encoding="utf-8")).get("inference_speed", {})
    write_reports(cfg, run_dir, preds, baseline, speed)


# ----------------------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------------------
def preflight(cfg: Cfg, run_dir: Path) -> bool:
    import transformers, peft  # noqa: E401
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Install the CUDA build of PyTorch and check `nvidia-smi`.")
    props = torch.cuda.get_device_properties(0)
    bf16 = use_bf16()
    log(f"GPU: {torch.cuda.get_device_name(0)} | compute {props.major}.{props.minor} | "
        f"{props.total_memory / 2 ** 30:.1f} GB | bf16={'yes' if bf16 else 'no (using fp16)'}")
    log(f"torch {torch.__version__} | transformers {transformers.__version__} | peft {peft.__version__}")
    if props.major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    free_gb = shutil.disk_usage(run_dir).free / 2 ** 30
    log(f"Free disk space at output dir: {free_gb:.1f} GB")
    if free_gb < 5:
        log("WARNING: less than 5 GB free disk space.")
    if props.total_memory / 2 ** 30 < 10:
        log("Note: <10 GB VRAM. The script will auto-fall-back (grad-checkpointing / smaller batch) on OOM.")
    return bf16


def main() -> None:
    cfg = parse_args()
    run_dir = Path(cfg.out_root) / cfg.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass
    logfile = open(run_dir / "run.log", "a", buffering=1, encoding="utf-8")
    sys.stdout, sys.stderr = Tee(sys.__stdout__, logfile), Tee(sys.__stderr__, logfile)
    log(f"===== START {cfg.run_name} ===== argv: {' '.join(sys.argv[1:])}")
    dump_json(cfg.__dict__, run_dir / "run_config.json")

    if cfg.plots_only:
        regenerate_plots(cfg, run_dir)
        return

    adapter = Path(cfg.adapter) if cfg.adapter else run_dir / "final_adapter"
    if cfg.eval_only and (adapter / "adapter_config.json").exists():
        base_id = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8")).get("base_model_name_or_path")
        if base_id and base_id != cfg.model_id:
            log(f"Adapter was trained on {base_id}; using that instead of {cfg.model_id}.")
            cfg.model_id = base_id

    keep_awake()
    bf16 = preflight(cfg, run_dir)
    data = prepare_data(cfg, run_dir)
    (run_dir / "plots").mkdir(exist_ok=True)
    plot_label_lengths(data.label_lens, data.gen_cap, run_dir / "plots" / "00_label_length_distribution.png")

    if cfg.eval_only:
        run_evaluation(cfg, run_dir, data, adapter, bf16)
        return

    if (run_dir / "TRAINING_DONE").exists() and not cfg.benchmark:
        log("TRAINING_DONE marker found - skipping training (delete it or change --run-name to retrain).")
    else:
        train_with_recovery(cfg, run_dir, data, bf16)
        if cfg.benchmark:
            return
    cleanup_cuda()

    if not cfg.skip_eval:
        run_evaluation(cfg, run_dir, data, adapter, bf16)
    else:
        if (run_dir / "train_log_history.csv").exists():
            plot_training(run_dir / "train_log_history.csv", run_dir / "plots")
    log("===== ALL DONE =====")


if __name__ == "__main__":
    main()