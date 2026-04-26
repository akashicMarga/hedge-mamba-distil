#!/usr/bin/env python
"""Latency vs decode-length benchmark.

Whisper's encoder always outputs a fixed-size representation (1500 frames
for 30 s of audio) regardless of actual speech duration. The decoder is
what scales with output length. This benchmark sweeps max_new_tokens from
short to long to show where the SSM constant-per-step advantage kicks in
over attention's growing KV-cache.

Teacher: O(L²) total — each of L steps attends over all previous tokens.
Student: O(L)  total — each step has the same constant SSM state size.

Usage:
    python scripts/long_audio_bench.py [--device mps|cpu|cuda]
"""
import sys, time, warnings, logging, argparse
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.student.whisper_mamba import WhisperMambaStudent
from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator
from torch.utils.data import Subset, DataLoader

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="mps")
parser.add_argument("--ckpt",       default="checkpoints/whisper_mamba/whisper_mamba_final.pt")
parser.add_argument("--model_id",   default="openai/whisper-tiny")
parser.add_argument("--state_size", type=int, default=64)
parser.add_argument("--n_audio",    type=int, default=8,  help="Audio clips to average over per length point")
parser.add_argument("--warmup",     type=int, default=2)
args = parser.parse_args()

DEVICE = args.device

# Token sweep: short → Whisper max (448). Log-spaced for a nice curve.
TOKEN_STEPS = [10, 20, 40, 70, 100, 150, 200, 300, 448]

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...", flush=True)
teacher = WhisperForConditionalGeneration.from_pretrained(
    args.model_id, torch_dtype=torch.float32).to(DEVICE).eval()

student = WhisperMambaStudent(
    WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float32),
    state_size=args.state_size)
student.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location="cpu"))
student = student.to(DEVICE).eval()

processor = WhisperProcessor.from_pretrained(args.model_id)

# ── Data — pre-compute encoder outputs once ───────────────────────────────────
print("Loading data...", flush=True)
_, val_loader, _ = make_librispeech_loaders(
    model_id=args.model_id, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=1, max_label_length=448,
)
loader = DataLoader(
    Subset(val_loader.dataset, range(args.warmup + args.n_audio)),
    batch_size=1, shuffle=False, collate_fn=LibriSpeechCollator(), num_workers=0,
)
batches = list(loader)

print("Pre-computing encoder outputs...", flush=True)
enc_outs, feats_list = [], []
with torch.no_grad():
    for batch in batches:
        f = batch["input_features"].to(DEVICE)
        enc_outs.append(teacher.model.encoder(f))
        feats_list.append(f)

# Use only post-warmup clips for measurement
enc_outs  = enc_outs[args.warmup:]
feats_list = feats_list[args.warmup:]


def sync():
    if DEVICE == "mps":  torch.mps.synchronize()
    elif DEVICE == "cuda": torch.cuda.synchronize()


def measure_latency(model, n_tokens):
    """Average latency (ms) over all audio clips at a fixed max_new_tokens."""
    lats = []
    gen_kw = dict(
        language="en", task="transcribe",
        max_new_tokens=n_tokens,
        min_new_tokens=n_tokens,   # force exactly n_tokens so both models decode same L
        repetition_penalty=1.1,
    )
    for feats, enc_out in zip(feats_list, enc_outs):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            model.generate(feats, encoder_outputs=enc_out, **gen_kw)
        sync()
        lats.append((time.perf_counter() - t0) * 1000)
    return np.mean(lats), np.std(lats)


# ── Sweep ─────────────────────────────────────────────────────────────────────
print(f"\nSweeping decode length  ({args.n_audio} clips each, device={DEVICE})\n")
print(f"{'Tokens':>7}  {'Teacher ms':>12}  {'Student ms':>12}  {'Speedup':>9}")
print("─" * 48)

teacher_lats, student_lats, speedups = [], [], []

for n in TOKEN_STEPS:
    t_mean, t_std = measure_latency(teacher, n)
    s_mean, s_std = measure_latency(student, n)
    sp = t_mean / s_mean
    teacher_lats.append(t_mean)
    student_lats.append(s_mean)
    speedups.append(sp)
    print(f"{n:>7}  {t_mean:>9.0f}ms  {s_mean:>9.0f}ms  {sp:>8.2f}×")

# ── Summary table ─────────────────────────────────────────────────────────────
print(f"\n{'═'*55}")
print(f"  Speedup summary  (device={DEVICE}, state_size={args.state_size})")
print(f"{'═'*55}")
crossover = next((TOKEN_STEPS[i] for i, s in enumerate(speedups) if s >= 1.0), None)
if crossover:
    print(f"  SSM becomes faster at ≥ {crossover} tokens")
else:
    print(f"  SSM not yet faster over this range — try larger state_size or longer sequences")

peak_sp  = max(speedups)
peak_tok = TOKEN_STEPS[speedups.index(peak_sp)]
print(f"  Peak speedup: {peak_sp:.2f}× at {peak_tok} tokens")
print(f"{'═'*55}")

# ── ASCII plot of latency curves ───────────────────────────────────────────────
print(f"\nLatency curves (ms):")
print(f"  Tokens  |  Teacher  |  Student")
print(f"  --------+-----------+----------")
max_lat = max(max(teacher_lats), max(student_lats))
bar_w = 30
for n, t, s in zip(TOKEN_STEPS, teacher_lats, student_lats):
    t_bar = int(t / max_lat * bar_w)
    s_bar = int(s / max_lat * bar_w)
    print(f"  {n:>6}  | {'█'*t_bar:<{bar_w}} {t:>5.0f}ms")
    print(f"          | {'░'*s_bar:<{bar_w}} {s:>5.0f}ms")
    print(f"          |")

print(f"\n  █ = Teacher  ░ = Student")
