#!/usr/bin/env python -u
"""MLX training-step benchmark: Metal kernel vs Python loop (no PT bridge).

Runs Stage-2 style forward+backward through WhisperMambaMLX using cached
LibriSpeech batches.  Compares per-step wall time for both scan paths.

Usage:
  python scripts/mlx_train_bench.py
"""
import sys, time
sys.path.insert(0, ".")

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

MODEL_ID = "openai/whisper-tiny"
MLX_REPO = "mlx-community/whisper-tiny-mlx"
BATCH    = 4
N_STEPS  = 20
WARMUP   = 5

# ── Build student ──────────────────────────────────────────────────────────────
print("Building WhisperMambaMLX student...", flush=True)
from src.mlx.model import WhisperMambaMLX
student = WhisperMambaMLX(mlx_repo=MLX_REPO, state_size=64)
print(student.param_summary(), flush=True)
student.freeze_for_stage2()

# ── Load cached batches as MLX arrays ─────────────────────────────────────────
print("Loading data...", flush=True)
from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator
from src.mlx.data import pt_batch_to_mlx
from torch.utils.data import Subset, DataLoader

_, val_loader_pt, _ = make_librispeech_loaders(
    model_id=MODEL_ID, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=BATCH, max_label_length=448,
)
raw_loader = DataLoader(
    Subset(val_loader_pt.dataset, range((WARMUP + N_STEPS) * BATCH)),
    batch_size=BATCH, shuffle=False,
    collate_fn=LibriSpeechCollator(), num_workers=0,
)
batches = [pt_batch_to_mlx(b) for b in raw_loader][: WARMUP + N_STEPS]

# ── Training step ──────────────────────────────────────────────────────────────
_BOS = 50258
optimizer = optim.Adam(learning_rate=1e-4)

from src.mlx.loss import ce_loss

def loss_fn(model, mel, dec_in, labels):
    logits, _ = model(mel, dec_in)
    return ce_loss(logits, labels)

loss_and_grad_fn = nn.value_and_grad(student, loss_fn)

def train_step(batch):
    mel, labels = batch["mel"], batch["labels"]
    B, L = labels.shape
    bos    = mx.full((B, 1), _BOS, dtype=mx.int32)
    dec_in = mx.concatenate([bos, mx.clip(labels[:, :-1], 0, None)], axis=1)
    loss, grads = loss_and_grad_fn(student, mel, dec_in, labels)
    optimizer.update(student, grads)
    mx.eval(student.parameters(), optimizer.state, loss)
    return loss.item()

# ── Benchmark helper ───────────────────────────────────────────────────────────
def run_bench(label, force_loop: bool):
    import src.ops.selective_scan as ss
    ss.FORCE_MLX_LOOP = force_loop

    for b in batches[:WARMUP]:
        train_step(b)

    times, losses = [], []
    for b in batches[WARMUP:]:
        t0   = time.perf_counter()
        loss = train_step(b)
        times.append((time.perf_counter() - t0) * 1000)
        losses.append(loss)

    ss.FORCE_MLX_LOOP = False
    return np.mean(times), np.std(times), np.mean(losses)

# ── Run ────────────────────────────────────────────────────────────────────────
print(f"\nBenchmarking {N_STEPS} MLX training steps  (batch={BATCH})\n", flush=True)

print("  [1/2] Metal kernel (no PT bridge)...", flush=True)
metal_ms, metal_std, metal_loss = run_bench("Metal", force_loop=False)

print("  [2/2] Python loop  (MLX lazy eval)...", flush=True)
loop_ms,  loop_std,  loop_loss  = run_bench("Loop",  force_loop=True)

speedup = loop_ms / metal_ms

print(f"""
┌──────────────────────────────────────────────────────────────┐
│  MLX training step  (batch={BATCH}, {N_STEPS} steps, no PT bridge)       │
├───────────────────────────────────┬────────────┬─────────────┤
│  Scan backend                     │  ms / step │  loss       │
├───────────────────────────────────┼────────────┼─────────────┤
│  Python loop  (MLX lazy eval)     │ {loop_ms:>7.0f} ±{loop_std:>3.0f}  │  {loop_loss:.4f}      │
│  Metal kernel (zero-copy)         │ {metal_ms:>7.0f} ±{metal_std:>3.0f}  │  {metal_loss:.4f}      │
├───────────────────────────────────┼────────────┼─────────────┤
│  Speedup                          │ {speedup:>9.2f}× │             │
└───────────────────────────────────┴────────────┴─────────────┘

  Time saved per step : {loop_ms - metal_ms:.0f} ms
  Projected Stage-2 full run (~3.5 h baseline):
    {metal_ms / loop_ms * 3.5:.1f} h with Metal kernel
""", flush=True)
