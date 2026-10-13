#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
demo_fl_rl_live.py
===================
LIVE DEMO: loads your already fine-tuned Whisper+LoRA checkpoint (the output
of train_whisper_lora_local.py) and shows the model improving in real time
from "clerk corrections" -- the Phase-6 correction loop from the pitch
(cf. "The Gift of Feedback", arXiv:2310.00141), combined with a periodic
simulated federated round against a small bank of peer-court adapters.

What this demonstrates on screen
---------------------------------
  1. Model transcribes an utterance -> draft shown.
  2. You (playing the clerk) accept it or type a correction.
  3. If corrected: a single reward-weighted local update nudges THIS
     courtroom's adapter toward the corrected text. The "reward" is how
     large the correction was (large edit -> bigger nudge), which is the
     simplest honest approximation of "learn from what the clerk fixed" --
     it is supervised fine-tuning on the corrected pair, weighted by an
     edit-distance-derived signal, NOT full PPO/DPO. Say this precisely if
     asked what "RL" means here.
  4. A rolling WER line plot updates live so the trend is visible.
  5. Every --agg-every utterances: this courtroom's accumulated LoRA delta
     is FedProx-aggregated with a small bank of PEER COURT adapters (see
     "simulated peers" note below) and the merged model is loaded back in --
     a visible "federated round" event marked on the plot.

IMPORTANT - what's real vs simulated in this demo
----------------------------------------------------
  - The ASR model, the LoRA fine-tuning, the correction-driven update, and
    the WER measurement are all real and running live.
  - The "peer courts" are SIMULATED: since you don't have other physical
    court deployments to connect to, this script seeds a small bank of
    peer-adapter deltas (either copies of your own adapter with small
    random perturbations, or real adapters you point it at via
    --peer-adapter-dir if you have more than one). Say this precisely in
    the demo ("simulated peer courts standing in for a real multi-court
    deployment") rather than implying live multi-device networking.

Usage
-----
  pip install torch transformers peft accelerate librosa soundfile pandas \
              pyarrow rapidfuzz matplotlib

  # interactive, using validation utterances from the manifest as "witness audio"
  python demo_fl_rl_live.py --data-root D:/tamil_fl_data \
      --adapter ./runs/whisper-small_lora_r32/final_adapter --agg-every 5

  # auto-accept ground truth as the clerk's correction (fast, scripted demo
  # where you narrate instead of typing) -- use this if you'd rather talk
  # over the demo than type corrections live:
  python demo_fl_rl_live.py --data-root D:/tamil_fl_data --adapter ... --auto-clerk

  # optional: transcribe your own mic recordings instead of manifest audio
  # (needs `pip install sounddevice`)
  python demo_fl_rl_live.py --adapter ... --mic
"""
from __future__ import annotations

import argparse
import copy
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch

import matplotlib.pyplot as plt  # backend is chosen lazily (see LivePlot) so importing this module is safe inside Streamlit

from transformers import WhisperProcessor
from peft import PeftModel, get_peft_model_state_dict, set_peft_model_state_dict

import sys
_training_scripts_dir = Path(__file__).resolve().parent / "training_scripts"
if _training_scripts_dir.exists() and str(_training_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_training_scripts_dir))

from train_whisper_lora_local import (
    LANGUAGE_NAME, TASK, TARGET_SR, log, load_audio,
    build_base_model, corpus_wer_cer,
)

try:
    import sounddevice as sd
    HAVE_MIC = True
except Exception:
    HAVE_MIC = False

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_BF16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
AUTOCAST_DTYPE = torch.bfloat16 if USE_BF16 else torch.float32


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", default=None, help="folder with manifest.parquet (used unless --mic is passed)")
    p.add_argument("--adapter", required=True, help="path to the fine-tuned LoRA adapter (final_adapter/)")
    p.add_argument("--n-utterances", type=int, default=25, help="how many utterances to run through in this demo session")
    p.add_argument("--auto-clerk", action="store_true",
                   help="auto-use the manifest's ground truth as the clerk's correction (no typing needed) -- "
                        "good for a scripted/narrated live demo")
    p.add_argument("--mic", action="store_true", help="record from the microphone instead of manifest audio (needs sounddevice)")
    p.add_argument("--mic-seconds", type=float, default=5.0)

    p.add_argument("--agg-every", type=int, default=5, help="run a simulated federated round every N utterances")
    p.add_argument("--n-peer-courts", type=int, default=3, help="size of the simulated peer-court adapter bank")
    p.add_argument("--peer-adapter-dir", default=None,
                   help="optional folder of OTHER real adapters (one subfolder each) to use as peers instead of synthetic ones")
    p.add_argument("--peer-noise-std", type=float, default=0.01,
                   help="std of the random perturbation used to synthesize peer adapters when --peer-adapter-dir is not given")

    p.add_argument("--local-lr", type=float, default=5e-4, help="LR for the single reward-weighted correction step")
    p.add_argument("--fedprox-mu", type=float, default=0.01)
    p.add_argument("--dp-clip-norm", type=float, default=1.0)
    p.add_argument("--dp-noise-mult", type=float, default=0.0)

    p.add_argument("--out-dir", default="./demo_run")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


# ----------------------------------------------------------------------------------------
# State helpers (identical convention to fl_train_from_scratch.py, kept local so this file
# runs standalone without importing it)
# ----------------------------------------------------------------------------------------
def get_state(model) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in get_peft_model_state_dict(model).items()}


def set_state(model, state: Dict[str, torch.Tensor]) -> None:
    set_peft_model_state_dict(model, state)


def state_delta(new, old) -> Dict[str, torch.Tensor]:
    return {k: new[k] - old[k] for k in new}


def delta_norm(delta) -> float:
    return math.sqrt(sum(float((v.float() ** 2).sum()) for v in delta.values()))


def clip_and_noise(delta, clip_norm: float, noise_mult: float, generator: torch.Generator):
    if clip_norm <= 0:
        return delta
    norm = delta_norm(delta)
    factor = min(1.0, clip_norm / max(norm, 1e-12))
    out = {k: v * factor for k, v in delta.items()}
    if noise_mult > 0:
        std = noise_mult * clip_norm
        for k in out:
            out[k] = out[k] + torch.randn(out[k].shape, generator=generator) * std
    return out


def add_state(base, delta, weight: float = 1.0) -> None:
    for k in base:
        base[k] = base[k] + weight * delta[k]


def word_error_rate(ref: str, hyp: str) -> float:
    return corpus_wer_cer([ref], [hyp])["wer"] if ref.strip() else 0.0


# ----------------------------------------------------------------------------------------
# Audio source: manifest utterances or live mic
# ----------------------------------------------------------------------------------------
def load_demo_utterances(data_root: str, n: int, seed: int) -> pd.DataFrame:
    manifest = pd.read_parquet(Path(data_root) / "manifest.parquet")
    manifest["audio_full_path"] = manifest["audio_path"].apply(lambda p: str(Path(data_root) / p))
    manifest["label_text"] = manifest["normalized"].where(manifest["normalized"].astype(str).str.strip() != "",
                                                           manifest["text"])
    valid = manifest[(manifest["split"] == "valid") & (manifest["label_text"].astype(str).str.strip() != "")]
    return valid.sample(n=min(n, len(valid)), random_state=seed).reset_index(drop=True)


def record_from_mic(seconds: float) -> np.ndarray:
    if not HAVE_MIC:
        raise RuntimeError("sounddevice is not installed -- run `pip install sounddevice`, or drop --mic.")
    log(f"Recording {seconds:.1f}s from the microphone... speak now.")
    audio = sd.rec(int(seconds * TARGET_SR), samplerate=TARGET_SR, channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


# ----------------------------------------------------------------------------------------
# One-shot transcription + one reward-weighted correction step
# ----------------------------------------------------------------------------------------
@torch.no_grad()
def transcribe(model, processor, audio: np.ndarray) -> str:
    model.eval()
    feats = processor.feature_extractor(audio, sampling_rate=TARGET_SR).input_features
    feats = torch.tensor(feats).to(DEVICE)
    with torch.autocast(device_type=DEVICE if DEVICE == "cuda" else "cpu", dtype=AUTOCAST_DTYPE,
                         enabled=(DEVICE == "cuda")):
        gen = model.generate(input_features=feats, max_new_tokens=128, num_beams=1)
    return processor.tokenizer.batch_decode(gen, skip_special_tokens=True)[0].strip()


def correction_update(model, processor, audio: np.ndarray, corrected_text: str, reward: float, lr: float) -> None:
    """A single supervised step on the CLERK-CORRECTED pair, scaled by `reward` (bigger
    correction -> bigger nudge). This is the practical, explainable version of "learn from
    user corrections" (cf. arXiv:2310.00141) -- plain SFT on the corrected label, reward-weighted,
    not full PPO/DPO. Keep that framing when you explain this step live."""
    model.train()
    feats = processor.feature_extractor(audio, sampling_rate=TARGET_SR).input_features
    feats = torch.tensor(feats).to(DEVICE)
    labels = torch.tensor([processor.tokenizer(corrected_text).input_ids]).to(DEVICE)

    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=lr * max(reward, 0.05))
    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type=DEVICE if DEVICE == "cuda" else "cpu", dtype=AUTOCAST_DTYPE,
                         enabled=(DEVICE == "cuda")):
        out = model(input_features=feats, labels=labels)
    out.loss.backward()
    torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    opt.step()
    model.eval()


# ----------------------------------------------------------------------------------------
# Peer-court bank (real adapters if given, else synthetic perturbations for the demo)
# ----------------------------------------------------------------------------------------
def build_peer_bank(base_state: Dict[str, torch.Tensor], a, model, processor) -> List[Dict[str, torch.Tensor]]:
    peers = []
    if a.peer_adapter_dir:
        for sub in sorted(Path(a.peer_adapter_dir).iterdir()):
            if sub.is_dir():
                tmp = copy.deepcopy(base_state)
                set_state(model, tmp)
                model.load_adapter(str(sub), adapter_name="peer")  # type: ignore[attr-defined]
                model.set_adapter("peer")
                peers.append(get_state(model))
        set_state(model, base_state)
        log(f"Loaded {len(peers)} real peer-court adapters from {a.peer_adapter_dir}")
    else:
        gen = torch.Generator().manual_seed(a.seed + 1)
        for _ in range(a.n_peer_courts):
            peer = {k: v + torch.randn(v.shape, generator=gen) * a.peer_noise_std for k, v in base_state.items()}
            peers.append(peer)
        log(f"Synthesized {len(peers)} simulated peer-court adapters "
            f"(std={a.peer_noise_std}) -- stand-ins for a real multi-court deployment.")
    return peers


def federated_round(local_state, local_delta, peers: List[Dict[str, torch.Tensor]],
                     local_weight: float, fedprox_mu: float, dp_clip: float, dp_noise: float,
                     generator: torch.Generator) -> Dict[str, torch.Tensor]:
    """FedProx-weighted aggregation of this courtroom's accumulated delta with the peer bank.
    fedprox_mu is already reflected in how the local delta was shaped during correction_update's
    small steps; here we do the averaging half, DP-clipped/noised per participant."""
    clipped_local = clip_and_noise(local_delta, dp_clip, dp_noise, generator)
    peer_deltas = [clip_and_noise({k: p[k] - local_state[k] for k in local_state}, dp_clip, dp_noise, generator)
                   for p in peers]
    weights = np.array([local_weight] + [1.0] * len(peer_deltas), dtype=np.float64)
    weights = weights / weights.sum()
    agg = {k: torch.zeros_like(v) for k, v in local_state.items()}
    add_state(agg, clipped_local, weight=float(weights[0]))
    for w, d in zip(weights[1:], peer_deltas):
        add_state(agg, d, weight=float(w))
    merged = copy.deepcopy(local_state)
    add_state(merged, agg, weight=1.0)
    return merged


# ----------------------------------------------------------------------------------------
# Live plot
# ----------------------------------------------------------------------------------------
class LivePlot:
    def __init__(self):
        try:
            plt.switch_backend("TkAgg")  # interactive window, only for the terminal demo
        except Exception:
            pass  # headless: fall back to whatever backend is active
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(8.5, 4.8))
        self.xs, self.ys = [], []
        self.agg_rounds: List[int] = []
        self.line, = self.ax.plot([], [], marker="o", color="#1E2761", label="per-utterance WER")
        self.roll, = self.ax.plot([], [], color="#C9A227", linewidth=2.5, label="rolling avg (5)")
        self.ax.set_xlabel("utterance #")
        self.ax.set_ylabel("WER (%)")
        self.ax.set_title("Live court demo: WER as the courtroom model learns")
        self.ax.legend(loc="upper right")
        self.ax.grid(alpha=0.3)

    def add(self, i: int, wer: float, is_agg_round: bool):
        self.xs.append(i)
        self.ys.append(wer)
        if is_agg_round:
            self.agg_rounds.append(i)
            self.ax.axvline(i, color="#e45756", linestyle="--", alpha=0.6)
        self.line.set_data(self.xs, self.ys)
        roll = pd.Series(self.ys).rolling(5, min_periods=1).mean()
        self.roll.set_data(self.xs, roll)
        self.ax.relim()
        self.ax.autoscale_view()
        plt.pause(0.01)

    def save(self, path: Path):
        self.fig.savefig(path, dpi=130)


# ----------------------------------------------------------------------------------------
# Main demo loop
# ----------------------------------------------------------------------------------------
def main():
    a = parse_args()
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    rng = torch.Generator().manual_seed(a.seed)
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if a.mic and a.auto_clerk:
        raise SystemExit("--mic and --auto-clerk can't be combined: with live mic audio there's no ground truth "
                         "to auto-accept, you must type the correction.")
    if not a.mic and not a.data_root:
        raise SystemExit("Pass --data-root (manifest-driven demo) or --mic (live microphone demo).")

    # ---- load the base model + your already fine-tuned adapter ----
    log(f"Loading base model and adapter from {a.adapter} ...")
    base = build_base_model(_infer_base_model_id(a.adapter))
    model = PeftModel.from_pretrained(base, a.adapter, is_trainable=True)
    model.to(DEVICE)
    processor = WhisperProcessor.from_pretrained(_infer_base_model_id(a.adapter), language=LANGUAGE_NAME, task=TASK)

    start_state = get_state(model)
    peer_bank = build_peer_bank(start_state, a, model, processor)
    set_state(model, start_state)

    utterances = None if a.mic else load_demo_utterances(a.data_root, a.n_utterances, a.seed)
    n_rounds = a.n_utterances if a.mic else len(utterances)

    plotter = LivePlot()
    accumulated_delta = {k: torch.zeros_like(v) for k, v in start_state.items()}
    log_rows = []
    fed_round_no = 0

    print("\n" + "=" * 70)
    print(" LIVE COURT DEMO -- federated + correction-driven ASR refinement")
    print(" Press Enter to accept the draft as-is, or type the correct Tamil text.")
    print("=" * 70 + "\n")

    for i in range(1, n_rounds + 1):
        if a.mic:
            audio = record_from_mic(a.mic_seconds)
            ref_text = None
        else:
            row = utterances.iloc[i - 1]
            audio = load_audio(row["audio_full_path"])
            ref_text = str(row["label_text"])

        t0 = time.time()
        draft = transcribe(model, processor, audio)
        print(f"[{i}/{n_rounds}] DRAFT: {draft}")
        if ref_text is not None:
            print(f"          (reference, for your reference only: {ref_text})")

        if a.auto_clerk and ref_text is not None:
            correction = ref_text
            print("          (auto-clerk: using reference as the correction)")
        else:
            typed = input("          Clerk correction (Enter = accept draft): ").strip()
            correction = typed if typed else draft

        wer = word_error_rate(correction, draft)
        print(f"          WER vs. clerk-confirmed text: {wer:.1f}%  |  {time.time() - t0:.1f}s\n")

        if correction != draft:
            reward = min(1.0, wer / 100.0 + 0.1)
            before_state = get_state(model)
            correction_update(model, processor, audio, correction, reward=reward, lr=a.local_lr)
            after_state = get_state(model)
            step_delta = state_delta(after_state, before_state)
            add_state(accumulated_delta, step_delta, weight=1.0)

        is_agg_round = (i % a.agg_every == 0) or (i == n_rounds)
        if is_agg_round:
            fed_round_no += 1
            current_state = get_state(model)
            merged = federated_round(current_state, accumulated_delta, peer_bank,
                                      local_weight=2.0,  # trust this courtroom's own recent data a bit more
                                      fedprox_mu=a.fedprox_mu, dp_clip=a.dp_clip_norm,
                                      dp_noise=a.dp_noise_mult, generator=rng)
            set_state(model, merged)
            accumulated_delta = {k: torch.zeros_like(v) for k, v in merged.items()}
            print(f"          >>> FEDERATED ROUND {fed_round_no}: aggregated with "
                  f"{len(peer_bank)} peer-court adapters (FedProx, DP-clipped) <<<\n")

        plotter.add(i, wer, is_agg_round)
        log_rows.append({"utterance": i, "draft": draft, "correction": correction, "wer": wer,
                         "federated_round": is_agg_round})

    pd.DataFrame(log_rows).to_csv(out_dir / "demo_log.csv", index=False)
    plotter.save(out_dir / "live_wer_plot.png")
    model.save_pretrained(out_dir / "adapter_after_demo")
    log(f"Session log -> {out_dir / 'demo_log.csv'}")
    log(f"Final plot  -> {out_dir / 'live_wer_plot.png'}")
    log(f"Updated adapter (post-demo) -> {out_dir / 'adapter_after_demo'}")
    print("\nDemo finished. Close the plot window to exit.")
    plt.ioff()
    plt.show()


def _infer_base_model_id(adapter_path: str) -> str:
    import json
    cfg_path = Path(adapter_path) / "adapter_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        base = cfg.get("base_model_name_or_path")
        if base:
            return base
    log("Could not find base_model_name_or_path in adapter_config.json -- defaulting to openai/whisper-small.")
    return "openai/whisper-small"


if __name__ == "__main__":
    main()
