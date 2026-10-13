"""
Pure aggregation logic for the federated LoRA-adapter server. No network / disk I/O here,
so every function is unit-testable (see test_server.py).

What the server does with client updates
-----------------------------------------
1. Parses each client's delta (a safetensors blob of LoRA-weight differences, float32).
2. Validates it against the current adapter's schema (same keys, same shapes, all finite).
3. RE-CLIPS it to the L2 bound server-side. Clients are asked to clip, but the server never
   trusts that -- a buggy or malicious client must not be able to dominate a round.
4. Takes a weighted average (weight = min(num_samples, MAX_WEIGHT_SAMPLES)), so one court with
   a huge session cannot outvote everyone else.
5. Optionally adds Gaussian noise to the average (central DP, trusted-aggregator model).

PRIVACY NOTE (read before claiming anything in a report/slide):
  Steps 3 and 5 are the *mechanics* of DP-FedAvg, but this code has NO privacy accountant, so it
  does not produce an (epsilon, delta) guarantee. With millions of LoRA parameters and only a few
  clients per round, the noise needed for a meaningful epsilon is far larger than the signal.
  Treat NOISE_MULT as an experimental knob (leave it 0 for the demo) and do not report an epsilon
  unless you add proper accounting (e.g. Opacus / dp-accounting) and measure the utility cost.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence, Tuple

import numpy as np
from safetensors.numpy import load as st_load
from safetensors.numpy import save as st_save

State = Dict[str, np.ndarray]


class DeltaRejected(ValueError):
    """Raised when a submitted update is malformed or incompatible; message is safe to show the client."""


def load_state_bytes(raw: bytes) -> State:
    try:
        tensors = st_load(raw)
    except Exception as e:  # corrupt payload, bf16 (unsupported by numpy), etc.
        raise DeltaRejected(
            f"could not parse safetensors payload as float tensors ({type(e).__name__}: {e}). "
            f"Adapters/deltas must be saved as float32 (cast with .float() before saving)."
        )
    return {k: np.asarray(v, dtype=np.float32) for k, v in tensors.items()}


def dump_state_bytes(state: State) -> bytes:
    return st_save({k: np.ascontiguousarray(v, dtype=np.float32) for k, v in state.items()})


def validate_schema(delta: State, reference: State) -> None:
    missing = sorted(set(reference) - set(delta))
    extra = sorted(set(delta) - set(reference))
    if missing or extra:
        raise DeltaRejected(
            f"tensor names do not match the current adapter: {len(missing)} missing "
            f"(e.g. {missing[:2]}), {len(extra)} unexpected (e.g. {extra[:2]})."
        )
    for k, ref in reference.items():
        if delta[k].shape != ref.shape:
            raise DeltaRejected(f"shape mismatch for '{k}': got {delta[k].shape}, expected {ref.shape}.")
        if not np.isfinite(delta[k]).all():
            raise DeltaRejected(f"non-finite values (NaN/Inf) in '{k}'.")


def l2_norm(state: State) -> float:
    return math.sqrt(sum(float(np.square(v, dtype=np.float64).sum()) for v in state.values()))


def clip_state(state: State, clip_norm: float) -> Tuple[State, float]:
    """Scale the whole update so its global L2 norm is <= clip_norm. Returns (clipped, original_norm)."""
    norm = l2_norm(state)
    if clip_norm <= 0 or norm <= clip_norm:
        return state, norm
    scale = clip_norm / norm
    return {k: (v * scale).astype(np.float32) for k, v in state.items()}, norm


def aggregate(
    deltas: Sequence[State],
    sample_counts: Sequence[int],
    clip_norm: float,
    noise_mult: float = 0.0,
    max_weight_samples: int = 200,
    rng: np.random.Generator | None = None,
) -> Tuple[State, dict]:
    """Weighted average of clipped client deltas (+ optional central Gaussian noise)."""
    if not deltas:
        raise ValueError("aggregate() needs at least one delta")
    rng = rng or np.random.default_rng()

    clipped, orig_norms = [], []
    for d in deltas:
        c, n = clip_state(d, clip_norm)
        clipped.append(c)
        orig_norms.append(n)

    weights = np.array([min(max(int(n), 1), max_weight_samples) for n in sample_counts], dtype=np.float64)
    wsum = float(weights.sum())

    avg: State = {}
    for k in clipped[0]:
        acc = np.zeros_like(clipped[0][k], dtype=np.float64)
        for w, d in zip(weights, clipped):
            acc += w * d[k].astype(np.float64)
        avg[k] = (acc / wsum).astype(np.float32)

    noise_std = 0.0
    if noise_mult > 0 and clip_norm > 0:
        # Sensitivity of a weighted average to one client = clip_norm * max_weight / sum_weights.
        noise_std = noise_mult * clip_norm * float(weights.max()) / wsum
        for k in avg:
            avg[k] = avg[k] + rng.normal(0.0, noise_std, size=avg[k].shape).astype(np.float32)

    info = {
        "n_clients": len(deltas),
        "weights": weights.tolist(),
        "client_norms_before_clip": [round(n, 6) for n in orig_norms],
        "clip_norm": clip_norm,
        "clipped_clients": int(sum(n > clip_norm > 0 for n in orig_norms)),
        "noise_mult": noise_mult,
        "noise_std_per_coord": noise_std,
        "aggregate_norm": round(l2_norm(avg), 6),
    }
    return avg, info


def apply_update(base: State, avg_delta: State, server_lr: float = 1.0) -> State:
    return {k: (base[k] + server_lr * avg_delta[k]).astype(np.float32) for k in base}
