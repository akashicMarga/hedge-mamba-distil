#!/usr/bin/env python
"""Measure real training-step wall time with Metal kernel ON vs OFF.

Runs Stage-2 style forward+backward through the full WhisperMambaStudent
using cached LibriSpeech batches.  Reports per-step time so you can
compare directly against a future run with the Python-loop scan.
"""
import sys, time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
sys.path.insert(0, ".")

import torch
import numpy as np
from transformers import WhisperForConditionalGeneration

DEVICE    = "mps"
MODEL_ID  = "openai/whisper-tiny"
BATCH     = 4
N_STEPS   = 20
WARMUP    = 5

def sync():
    torch.mps.synchronize()

# ── Build student ─────────────────────────────────────────────────────────────
print("Building student model...")
from src.student.whisper_mamba import WhisperMambaStudent
from src.ops.selective_scan import _try_build_kernel
_try_build_kernel(128)

base = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
student = WhisperMambaStudent(base, state_size=64).to(DEVICE).float()

# Stage-2 trainable params (SSM, cross-attn, FFN, layer norms)
for name, p in student.named_parameters():
    layer_types = ("self_attn", "encoder_attn", "fc1", "fc2",
                   "self_attn_layer_norm", "encoder_attn_layer_norm",
                   "final_layer_norm")
    p.requires_grad = any(k in name for k in layer_types)

optimizer = torch.optim.AdamW(
    [p for p in student.parameters() if p.requires_grad], lr=1e-4)

# ── Load a few cached batches ─────────────────────────────────────────────────
print("Loading data...")
from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator
from torch.utils.data import Subset, DataLoader

_, val_loader, _ = make_librispeech_loaders(
    model_id=MODEL_ID, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=BATCH, max_label_length=448,
)
loader = DataLoader(
    Subset(val_loader.dataset, range((WARMUP + N_STEPS) * BATCH)),
    batch_size=BATCH, shuffle=False,
    collate_fn=LibriSpeechCollator(), num_workers=0,
)
batches = list(loader)[:WARMUP + N_STEPS]

# ── One training step ─────────────────────────────────────────────────────────
BOS_ID = 50258   # Whisper BOS token

def train_step(batch):
    feats  = batch["input_features"].to(DEVICE)
    labels = batch["labels"].to(DEVICE)
    # Build decoder_input_ids: [BOS, label_0, ..., label_{L-2}]
    bos    = torch.full((feats.shape[0], 1), BOS_ID, device=DEVICE, dtype=torch.long)
    dec_in = torch.cat([bos, labels[:, :-1].clamp(min=0)], dim=1)

    optimizer.zero_grad()
    out  = student(feats, dec_in, labels=labels)
    loss = out.loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in student.parameters() if p.requires_grad], 1.0)
    optimizer.step()
    return loss.item()

# ── Benchmark helper ──────────────────────────────────────────────────────────
def run_bench(label, use_metal):
    import src.ops.selective_scan as ss_mod
    # Patch the use_metal default inside selective_scan
    original_forward = ss_mod.selective_scan

    if not use_metal:
        import functools
        ss_mod.selective_scan = functools.partial(original_forward, use_metal=False)

    # Warmup
    student.train()
    for b in batches[:WARMUP]:
        train_step(b)
        sync()

    # Timed steps
    losses, times = [], []
    for b in batches[WARMUP:]:
        sync()
        t0 = time.perf_counter()
        loss = train_step(b)
        sync()
        times.append((time.perf_counter() - t0) * 1000)
        losses.append(loss)

    if not use_metal:
        ss_mod.selective_scan = original_forward

    return np.mean(times), np.std(times), np.mean(losses)

# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nBenchmarking {N_STEPS} training steps  (batch={BATCH}, device={DEVICE})\n")

print(f"  [1/2] Metal kernel (use_metal=True)...")
metal_ms, metal_std, metal_loss = run_bench("Metal", use_metal=True)

print(f"  [2/2] Python loop  (use_metal=False)...")
loop_ms,  loop_std,  loop_loss  = run_bench("Loop",  use_metal=False)

speedup = loop_ms / metal_ms

print(f"""
┌──────────────────────────────────────────────────────────────┐
│  Training step time  (batch={BATCH}, {N_STEPS} steps, device={DEVICE})     │
├───────────────────────────────────┬────────────┬─────────────┤
│  Scan backend                     │  ms / step │  loss       │
├───────────────────────────────────┼────────────┼─────────────┤
│  Python loop  (original)          │ {loop_ms:>7.0f} ±{loop_std:>3.0f}  │  {loop_loss:.4f}      │
│  Metal kernel (new)               │ {metal_ms:>7.0f} ±{metal_std:>3.0f}  │  {metal_loss:.4f}      │
├───────────────────────────────────┼────────────┼─────────────┤
│  Speedup                          │ {speedup:>9.2f}× │             │
└───────────────────────────────────┴────────────┴─────────────┘

  Time saved per step:  {loop_ms - metal_ms:.0f} ms
  Projected for 500M-token Stage-2 run (~6h baseline):
    Scan fraction of step ≈ {loop_ms/loop_ms*100:.0f}% → {loop_ms/loop_ms*6:.1f}h → {metal_ms/loop_ms*6:.1f}h with Metal
""")
