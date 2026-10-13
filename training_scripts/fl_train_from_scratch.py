#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fl_train_from_scratch.py
=========================
Train Whisper-small for Tamil ASR ENTIRELY via (simulated) Federated Learning.
No pre-existing checkpoint is used or required -- the model starts from the
public `openai/whisper-small` weights and only ever learns through federated
rounds. This is the "core contribution" script: district = FL client,
LoRA-only local updates, IFCA-style clustering, FedProx-weighted aggregation,
DP-noised updates.

IMPORTANT - what "federated" means here
----------------------------------------
This is a *simulation* of FL on one machine (the standard way FL is
studied/demoed in research -- Flower's own `start_simulation`, FedML's
single-process mode, and most FL papers' experiments all work the same way).
Each "client" is a Python object holding its own LoRA weights; nothing is
pooled across clients when computing a gradient; only LoRA-weight deltas are
ever "transmitted" (i.e. passed between in-memory objects / written to disk).
Say this precisely if asked -- it is not literally multiple physical devices
exchanging weights over a network, and that's fine; real FL deployments are
written this way too, just with a network call where this script has a
Python function call.

Reuses from train_whisper_lora_local.py (must be in the same folder):
  - the manifest / dataset / collator pipeline
  - WER/CER scoring (corpus_wer_cer)
  - the base-model loader (build_base_model)

Usage
-----
  pip install torch transformers peft accelerate librosa soundfile pandas \
              pyarrow rapidfuzz scikit-learn matplotlib

  python fl_train_from_scratch.py --data-root D:/tamil_fl_data \
      --rounds 40 --clients-per-round 6 --clusters 3 --local-steps 60

Output (in --out-root/--run-name/):
  cluster_adapters/cluster_XX/      -> final LoRA adapter for each dialect cluster
  global_adapter/                   -> a single deployable adapter (data-weighted
                                        average across clusters) for convenience
  round_log.csv                     -> round-by-round client/cluster/WER history
  plots/fl_progress.png             -> WER per cluster over rounds
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import WhisperProcessor
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

# ---- reuse the existing, battle-tested pipeline instead of rewriting it ----
from train_whisper_lora_local import (
    LANGUAGE_NAME, TASK, TARGET_SR, log,
    TamilASRDataset, DataCollatorSpeechSeq2SeqWithPadding,
    corpus_wer_cer, build_base_model,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32


# ----------------------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", required=True, help="folder containing manifest.parquet + audio (same as train_whisper_lora_local.py)")
    p.add_argument("--out-root", default="./fl_runs")
    p.add_argument("--run-name", default="fl_from_scratch")
    p.add_argument("--model", default="openai/whisper-small")

    p.add_argument("--rounds", type=int, default=40, help="number of federated communication rounds")
    p.add_argument("--clients-per-round", type=int, default=6, help="districts sampled each round")
    p.add_argument("--local-steps", type=int, default=60, help="local SGD steps per client per round")
    p.add_argument("--local-batch-size", type=int, default=4)
    p.add_argument("--local-lr", type=float, default=1e-3)

    p.add_argument("--clusters", type=int, default=3, help="IFCA cluster count (dialect groups)")
    p.add_argument("--recluster-every", type=int, default=3, help="re-run IFCA cluster assignment every N rounds")

    p.add_argument("--fedprox-mu", type=float, default=0.01, help="FedProx proximal term strength (0 = plain FedAvg)")

    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--lora-targets", choices=["qv", "all"], default="qv")

    p.add_argument("--dp-clip-norm", type=float, default=1.0, help="L2 clip applied to each client's LoRA delta before aggregation (0 = off)")
    p.add_argument("--dp-noise-mult", type=float, default=0.0, help="Gaussian noise multiplier (std = mult * clip_norm). 0 = no DP noise, try 0.3-1.0 for a real epsilon")

    p.add_argument("--min-client-hours", type=float, default=0.2, help="skip districts with less train audio than this")
    p.add_argument("--max-client-hours", type=float, default=3.0, help="cap per-district train audio used per round (keeps rounds fast)")
    p.add_argument("--eval-every", type=int, default=4, help="evaluate WER every N rounds")
    p.add_argument("--eval-subset", type=int, default=150, help="valid utterances used per quick WER check")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ----------------------------------------------------------------------------------------
# LoRA (de)serialization helpers
# ----------------------------------------------------------------------------------------
def build_lora_model(model_id: str, r: int, alpha: int, dropout: float, targets: List[str]):
    model = build_base_model(model_id)
    lora = LoraConfig(r=r, lora_alpha=alpha, target_modules=targets, lora_dropout=dropout, bias="none")
    model = get_peft_model(model, lora)
    return model


def get_state(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def set_state(model, state: Dict[str, torch.Tensor]) -> None:
    set_peft_model_state_dict(model, state)


def state_delta(new: Dict[str, torch.Tensor], old: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: new[k] - old[k] for k in new}


def delta_norm(delta: Dict[str, torch.Tensor]) -> float:
    return math.sqrt(sum(float((v.float() ** 2).sum()) for v in delta.values()))


def clip_and_noise(delta: Dict[str, torch.Tensor], clip_norm: float, noise_mult: float,
                    generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """DP-SGD-style mechanism (Abadi et al., 2016) applied to a federated client update:
    clip the whole update to an L2 ball, then add calibrated Gaussian noise. Operating on the
    LoRA delta (a few MB) rather than the full model makes this cheap enough to run every round."""
    if clip_norm <= 0:
        return delta
    norm = delta_norm(delta)
    factor = min(1.0, clip_norm / max(norm, 1e-12))
    out = {k: v * factor for k, v in delta.items()}
    if noise_mult > 0:
        std = noise_mult * clip_norm
        for k, v in out.items():
            out[k] = v + torch.randn(v.shape, generator=generator) * std
    return out


def add_state(base: Dict[str, torch.Tensor], delta: Dict[str, torch.Tensor], weight: float = 1.0) -> None:
    for k in base:
        base[k] = base[k] + weight * delta[k]


def zeros_like_state(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: torch.zeros_like(v) for k, v in state.items()}


def flatten(state: Dict[str, torch.Tensor]) -> np.ndarray:
    return torch.cat([v.flatten().float() for v in state.values()]).numpy()


# ----------------------------------------------------------------------------------------
# Local client training (the only place that ever touches a client's raw audio)
# ----------------------------------------------------------------------------------------
def local_train(model, processor, client_df: pd.DataFrame, global_state: Dict[str, torch.Tensor],
                 steps: int, batch_size: int, lr: float, fedprox_mu: float, bad_log: str) -> Dict[str, torch.Tensor]:
    model.to(DEVICE)
    model.train()
    ds = TamilASRDataset(client_df, processor, bad_log=bad_log)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    loader = DataLoader(ds, batch_size=min(batch_size, len(ds)), shuffle=True, collate_fn=collator, drop_last=False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr)
    global_gpu = {k: v.to(DEVICE) for k, v in global_state.items()} if fedprox_mu > 0 else None

    it = iter(loader)
    done = 0
    while done < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE if DEVICE == "cuda" else "cpu", dtype=AUTOCAST_DTYPE,
                             enabled=(DEVICE == "cuda")):
            out = model(input_features=batch["input_features"], labels=batch["labels"])
            loss = out.loss
            if fedprox_mu > 0:
                prox = 0.0
                local_state = get_peft_model_state_dict(model)
                for k, v in local_state.items():
                    prox = prox + ((v - global_gpu[k]) ** 2).sum()
                loss = loss + (fedprox_mu / 2.0) * prox
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        opt.step()
        done += 1
    model.eval()
    return get_state(model)


# ----------------------------------------------------------------------------------------
# Quick WER check (used for both IFCA cluster-assignment and periodic reporting)
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def quick_wer(model, processor, df: pd.DataFrame, gen_max_len: int = 128) -> float:
    model.to(DEVICE)
    model.eval()
    ds = TamilASRDataset(df, processor)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    loader = DataLoader(ds, batch_size=8, shuffle=False, collate_fn=collator)
    refs, hyps = [], []
    for batch in loader:
        feats = batch["input_features"].to(DEVICE)
        with torch.autocast(device_type=DEVICE if DEVICE == "cuda" else "cpu", dtype=AUTOCAST_DTYPE,
                             enabled=(DEVICE == "cuda")):
            gen = model.generate(input_features=feats, max_new_tokens=gen_max_len, num_beams=1)
        hyp = processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        lab = batch["labels"].clone()
        lab[lab == -100] = processor.tokenizer.pad_token_id
        ref = processor.tokenizer.batch_decode(lab, skip_special_tokens=True)
        refs += ref
        hyps += hyp
    return corpus_wer_cer(refs, hyps)["wer"]


@torch.no_grad()
def quick_loss(model, processor, df: pd.DataFrame, n_batches: int = 4, batch_size: int = 8) -> float:
    """Cheap forward-only loss, used to decide which IFCA cluster a client fits best."""
    model.to(DEVICE)
    model.eval()
    sub = df.sample(n=min(len(df), n_batches * batch_size), random_state=0) if len(df) > 0 else df
    ds = TamilASRDataset(sub, processor)
    collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collator)
    total, n = 0.0, 0
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast(device_type=DEVICE if DEVICE == "cuda" else "cpu", dtype=AUTOCAST_DTYPE,
                             enabled=(DEVICE == "cuda")):
            out = model(input_features=batch["input_features"], labels=batch["labels"])
        total += float(out.loss) * batch["input_features"].shape[0]
        n += batch["input_features"].shape[0]
    return total / max(n, 1)


# ----------------------------------------------------------------------------------------
# Data partition: district == FL client (mirrors the pitch's core framing)
# ----------------------------------------------------------------------------------------
def build_clients(manifest: pd.DataFrame, min_hours: float, max_hours: float) -> Dict[str, pd.DataFrame]:
    train = manifest[manifest["split"] == "train"]
    clients = {}
    for district, g in train.groupby("district", dropna=False):
        hours = g["duration"].sum() / 3600.0
        if hours < min_hours:
            continue
        g = g.sample(frac=1.0, random_state=0)
        kept = g[g["duration"].cumsum() / 3600 <= max_hours]
        clients[str(district)] = kept.reset_index(drop=True)
    log(f"Built {len(clients)} FL clients (districts) after the {min_hours}-{max_hours}h/district window.")
    return clients


# ----------------------------------------------------------------------------------------
# Main FL loop: IFCA (cluster assignment) + FedProx (local objective) + FedAvg-style
# weighted aggregation within each cluster, with DP-noised client updates.
# ----------------------------------------------------------------------------------------
def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = torch.Generator().manual_seed(a.seed)

    run_dir = Path(a.out_root) / a.run_name
    (run_dir / "cluster_adapters").mkdir(parents=True, exist_ok=True)
    (run_dir / "plots").mkdir(exist_ok=True)

    manifest = pd.read_parquet(Path(a.data_root) / "manifest.parquet")
    manifest["audio_full_path"] = manifest["audio_path"].apply(lambda p: str(Path(a.data_root) / p))
    manifest["label_text"] = manifest["normalized"].where(manifest["normalized"].astype(str).str.strip() != "",
                                                           manifest["text"])
    manifest = manifest[manifest["label_text"].astype(str).str.strip() != ""].reset_index(drop=True)

    processor = WhisperProcessor.from_pretrained(a.model, language=LANGUAGE_NAME, task=TASK)
    targets = ["q_proj", "v_proj"] if a.lora_targets == "qv" else \
        ["q_proj", "k_proj", "v_proj", "out_proj", "fc1", "fc2"]

    clients = build_clients(manifest, a.min_client_hours, a.max_client_hours)
    client_ids = list(clients.keys())
    if len(client_ids) < a.clusters:
        raise ValueError(f"Only {len(client_ids)} usable districts but --clusters={a.clusters}. Lower --clusters "
                         f"or --min-client-hours.")

    valid_df = manifest[manifest["split"] == "valid"]
    eval_subset = valid_df.sample(n=min(a.eval_subset, len(valid_df)), random_state=1).reset_index(drop=True)
    # per-cluster eval slices (helps show *why* clustering matters): split eval set round-robin
    cluster_eval = {c: eval_subset.iloc[c::a.clusters].reset_index(drop=True) for c in range(a.clusters)}

    # ---- init: one model object we keep re-using across clients/clusters (swap LoRA state in/out)
    log("Building base model + LoRA scaffold (weights start from public Whisper-small checkpoint)...")
    work_model = build_lora_model(a.model, a.lora_r, a.lora_alpha, a.lora_dropout, targets)
    init_state = get_state(work_model)

    # start all clusters from the SAME random/base LoRA init (true "from scratch")
    cluster_states = {c: copy.deepcopy(init_state) for c in range(a.clusters)}
    client_cluster = {cid: random.randrange(a.clusters) for cid in client_ids}  # random init assignment

    bad_log = str(run_dir / "bad_audio.txt")
    history = []

    for rnd in range(1, a.rounds + 1):
        t0 = time.time()
        selected = random.sample(client_ids, k=min(a.clients_per_round, len(client_ids)))

        # ---- IFCA: periodically re-estimate which cluster each selected client best fits ----
        if rnd == 1 or rnd % a.recluster_every == 0:
            for cid in selected:
                losses = []
                for c in range(a.clusters):
                    set_state(work_model, cluster_states[c])
                    losses.append(quick_loss(work_model, processor, clients[cid], n_batches=2, batch_size=a.local_batch_size))
                client_cluster[cid] = int(np.argmin(losses))

        # ---- local training per selected client, starting from its assigned cluster's weights ----
        cluster_deltas: Dict[int, List[Dict[str, torch.Tensor]]] = {c: [] for c in range(a.clusters)}
        cluster_weights: Dict[int, List[float]] = {c: [] for c in range(a.clusters)}

        for cid in selected:
            c = client_cluster[cid]
            set_state(work_model, cluster_states[c])
            local_state = local_train(work_model, processor, clients[cid], cluster_states[c],
                                       steps=a.local_steps, batch_size=a.local_batch_size,
                                       lr=a.local_lr, fedprox_mu=a.fedprox_mu, bad_log=bad_log)
            delta = state_delta(local_state, cluster_states[c])
            delta = clip_and_noise(delta, a.dp_clip_norm, a.dp_noise_mult, rng)
            cluster_deltas[c].append(delta)
            cluster_weights[c].append(float(clients[cid]["duration"].sum()))

        # ---- FedAvg-style weighted aggregation within each cluster (the "Prox" already shaped
        #      the local objective; this step is the averaging half of FedProx) ----
        for c in range(a.clusters):
            if not cluster_deltas[c]:
                continue
            w = np.array(cluster_weights[c], dtype=np.float64)
            w = w / w.sum()
            agg = zeros_like_state(cluster_states[c])
            for wi, d in zip(w, cluster_deltas[c]):
                add_state(agg, d, weight=float(wi))
            add_state(cluster_states[c], agg, weight=1.0)

        dt = time.time() - t0
        row = {"round": rnd, "clients": ",".join(selected),
               "cluster_sizes": json.dumps({c: sum(1 for cid in selected if client_cluster[cid] == c)
                                             for c in range(a.clusters)}),
               "seconds": round(dt, 1)}

        if rnd % a.eval_every == 0 or rnd == a.rounds:
            for c in range(a.clusters):
                set_state(work_model, cluster_states[c])
                row[f"wer_cluster{c}"] = quick_wer(work_model, processor, cluster_eval[c])
            log(f"[round {rnd}/{a.rounds}] " +
                " | ".join(f"cluster{c} WER {row.get(f'wer_cluster{c}', float('nan')):.1f}" for c in range(a.clusters)) +
                f" | {dt:.0f}s")
        else:
            log(f"[round {rnd}/{a.rounds}] trained {len(selected)} clients, {dt:.0f}s "
                f"(cluster sizes this round: {row['cluster_sizes']})")
        history.append(row)

        # periodic checkpoint so a crash doesn't lose progress
        if rnd % max(1, a.eval_every) == 0 or rnd == a.rounds:
            for c in range(a.clusters):
                set_state(work_model, cluster_states[c])
                work_model.save_pretrained(run_dir / "cluster_adapters" / f"cluster_{c:02d}")
            pd.DataFrame(history).to_csv(run_dir / "round_log.csv", index=False)

    # ---- final deployable single adapter: data-weighted average across clusters ----
    sizes = {c: sum(clients[cid]["duration"].sum() for cid in client_ids if client_cluster.get(cid) == c)
             for c in range(a.clusters)}
    total = sum(sizes.values()) or 1.0
    global_state = zeros_like_state(init_state)
    for c in range(a.clusters):
        add_state(global_state, cluster_states[c], weight=sizes[c] / total)
    set_state(work_model, global_state)
    work_model.save_pretrained(run_dir / "global_adapter")
    log(f"Saved per-cluster adapters -> {run_dir / 'cluster_adapters'}")
    log(f"Saved a single deployable (cluster-weighted) adapter -> {run_dir / 'global_adapter'}")

    # ---- plot ----
    hist = pd.DataFrame(history)
    wer_cols = [c for c in hist.columns if c.startswith("wer_cluster")]
    if wer_cols:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        for col in wer_cols:
            sub = hist[["round", col]].dropna()
            ax.plot(sub["round"], sub[col], marker="o", label=col.replace("wer_", ""))
        ax.set_xlabel("FL round")
        ax.set_ylabel("WER (%)")
        ax.set_title("Per-cluster WER over federated rounds (training from scratch)")
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(run_dir / "plots" / "fl_progress.png", dpi=130)
        log(f"Saved progress plot -> {run_dir / 'plots' / 'fl_progress.png'}")

    log("===== FL TRAINING FROM SCRATCH: DONE =====")


if __name__ == "__main__":
    main()
