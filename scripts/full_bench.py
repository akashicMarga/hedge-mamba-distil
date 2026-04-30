#!/usr/bin/env python
"""Full benchmark: inference & training-scan across all backends.

Columns
───────
  A. PyTorch teacher   — Whisper-tiny, multi-head attention, KV-cache
  B. PyTorch student   — HedgeMamba, Python-loop scan, h/conv-cache
  C. PyTorch student + Metal kernel  — same model, Metal scan (training path)
  D. MLX teacher       — mlx-community/whisper-tiny-mlx
  E. MLX student       — HedgeMamba MLX, lazy-eval fused scan

Notes:
  - Inference (A/B/D/E): autoregressive decoder, L=1 per step → Metal kernel
    is bypassed (overhead > benefit at L=1).  Student advantage comes from
    O(1) h_cache vs O(L) KV-cache in the teacher.
  - Metal kernel (C): measured on the TRAINING forward path, where the entire
    target sequence (L up to 448) is scanned in one shot (teacher-forcing).
    Not applicable to autoregressive inference.
"""

import sys, time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
sys.path.insert(0, ".")

import torch
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor

DEVICE    = "mps"
CKPT      = "checkpoints/whisper_mamba/whisper_mamba_final.pt"
MODEL_ID  = "openai/whisper-tiny"
MLX_REPO  = "mlx-community/whisper-tiny-mlx"
N_SAMPLES = 20
WARMUP    = 3

GEN_KWARGS = dict(language="en", task="transcribe",
                  max_new_tokens=128, repetition_penalty=1.1)

def sync():
    torch.mps.synchronize()


# ─────────────────────────────────────────────────────────────────
# SECTION 1: Inference benchmark (A, B, D, E)
# ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  Loading models")
print("=" * 60)

# A – PyTorch teacher
print("  [A] PyTorch teacher (Whisper-tiny)...")
teacher_pt = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32).to(DEVICE).eval()
processor  = WhisperProcessor.from_pretrained(MODEL_ID)

# B/C – PyTorch student (Metal scan is transparent; inference bypasses it anyway)
print("  [B] PyTorch student (HedgeMamba)...")
from src.student.whisper_mamba import WhisperMambaStudent
from src.ops.selective_scan import _try_build_kernel
_try_build_kernel(128)   # pre-compile Metal kernel

student_pt = WhisperMambaStudent(
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32),
    state_size=64)
student_pt.load_state_dict(torch.load(CKPT, weights_only=True, map_location="cpu"))
student_pt = student_pt.to(DEVICE).eval()

# D – MLX teacher
print("  [D] MLX teacher (mlx-community/whisper-tiny-mlx)...")
import mlx.core as mx
import mlx_whisper.load_models as lm
import mlx_whisper.decoding as dec
mlx_teacher = lm.load_model(MLX_REPO)
_GEN_OPTS = dec.DecodingOptions(
    language="en", task="transcribe", fp16=False,
    without_timestamps=True, suppress_blank=True)

# E – MLX student
print("  [E] MLX student (HedgeMamba MLX)...")
from scripts.mlx_inference import build_mlx_student
pt_ckpt    = torch.load(CKPT, weights_only=True, map_location="cpu")
mlx_student = build_mlx_student(pt_ckpt, MLX_REPO, state_size=64)

# Synthetic data: N_SAMPLES + WARMUP batches of (1, 80, 3000) mel
print("\n  Generating synthetic audio inputs...")
mels_pt = [torch.randn(1, 80, 3000).float() for _ in range(N_SAMPLES + WARMUP)]

def bench_pt(model, name, mels):
    lats, n_toks = [], []
    for i, mel in enumerate(mels):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(mel.to(DEVICE), **GEN_KWARGS)
        sync()
        ms = (time.perf_counter() - t0) * 1000
        if i >= WARMUP:
            lats.append(ms)
            n_toks.append(ids.shape[-1])
        if i == WARMUP:
            print(f"    {name}: first timed sample = {ms:.0f} ms", flush=True)
    return np.mean(lats), sum(n_toks) / (sum(lats) / 1000)

def bench_mlx(model, name, mels):
    lats, n_toks = [], []
    for i, mel in enumerate(mels):
        feats = mx.array(mel.squeeze(0).numpy().T[None])
        t0 = time.perf_counter()
        result = dec.decode(model, feats, _GEN_OPTS)
        ms = (time.perf_counter() - t0) * 1000
        if i >= WARMUP:
            lats.append(ms)
            n_toks.append(len(result[0].tokens))
        if i == WARMUP:
            print(f"    {name}: first timed sample = {ms:.0f} ms", flush=True)
    return np.mean(lats), sum(n_toks) / (sum(lats) / 1000)

print("\n" + "=" * 60)
print("  SECTION 1 — Inference (autoregressive decode)")
print("  (Metal scan kernel NOT active here — L=1 per step)")
print("=" * 60)

a_lat, a_tps = bench_pt(teacher_pt,  "[A] PT teacher", mels_pt)
b_lat, b_tps = bench_pt(student_pt,  "[B] PT student", mels_pt)
d_lat, d_tps = bench_mlx(mlx_teacher, "[D] MLX teacher", mels_pt)
e_lat, e_tps = bench_mlx(mlx_student, "[E] MLX student", mels_pt)

print(f"""
┌─────────────────────────────────────────────────────────────┐
│  Inference — autoregressive decoder  ({N_SAMPLES} samples)        │
├───────────────────────────────────────┬────────────┬────────┤
│  Backend                              │  Lat (ms)  │  tok/s │
├───────────────────────────────────────┼────────────┼────────┤
│  [A] PyTorch teacher  (MPS, attn+KV)  │ {a_lat:>9.0f}  │ {a_tps:>6.1f} │
│  [B] PyTorch student  (MPS, h_cache)  │ {b_lat:>9.0f}  │ {b_tps:>6.1f} │
│  [D] MLX teacher      (lazy Metal)    │ {d_lat:>9.0f}  │ {d_tps:>6.1f} │
│  [E] MLX student      (lazy Metal)    │ {e_lat:>9.0f}  │ {e_tps:>6.1f} │
├───────────────────────────────────────┼────────────┼────────┤
│  PT student vs PT teacher             │ {a_lat/b_lat:>9.2f}× │        │
│  MLX student vs MLX teacher           │ {d_lat/e_lat:>9.2f}× │        │
│  MLX student vs PT student            │ {b_lat/e_lat:>9.2f}× │        │
└───────────────────────────────────────┴────────────┴────────┘
""")


# ─────────────────────────────────────────────────────────────────
# SECTION 2: Training-scan benchmark (B vs C)
#   Teacher-forcing mode: full sequence scan over L=448 tokens
# ─────────────────────────────────────────────────────────────────

print("=" * 60)
print("  SECTION 2 — Training scan (teacher-forcing, L=448)")
print("  (This IS where the Metal kernel fires)")
print("=" * 60)

from src.ops.selective_scan import selective_scan
import src.ops.selective_scan as ss_mod

B_b, L, D2, Ns = 2, 448, 768, 128
u  = torch.randn(B_b, L, D2, device=DEVICE).float()
dt = (torch.rand(B_b, L, D2, device=DEVICE)*0.09+0.01).float()
A  = (-torch.arange(1, Ns+1, device=DEVICE).float()).unsqueeze(0).expand(D2, -1)
Bf = torch.randn(B_b, L, Ns, device=DEVICE).float()
Cf = torch.randn(B_b, L, Ns, device=DEVICE).float()
h0 = torch.zeros(B_b, D2, Ns, device=DEVICE)

N_SCAN = 20

# Warmup both paths
for _ in range(5):
    selective_scan(u, dt, A, Bf, Cf, h0, use_metal=True);  sync()
    selective_scan(u, dt, A, Bf, Cf, h0, use_metal=False); sync()

t0 = time.perf_counter()
for _ in range(N_SCAN):
    selective_scan(u, dt, A, Bf, Cf, h0, use_metal=False); sync()
lat_py = (time.perf_counter()-t0)/N_SCAN*1000

t0 = time.perf_counter()
for _ in range(N_SCAN):
    selective_scan(u, dt, A, Bf, Cf, h0, use_metal=True); sync()
lat_metal = (time.perf_counter()-t0)/N_SCAN*1000

# MLX scan (pure MLX, no PT round-trip)
u_mx  = mx.array(u.cpu().numpy())
dt_mx = mx.array(dt.cpu().numpy())
A_mx  = mx.array(A.cpu().numpy())
Bf_mx = mx.array(Bf.cpu().numpy())
Cf_mx = mx.array(Cf.cpu().numpy())
h0_mx = mx.array(h0.cpu().numpy())

from src.student.mlx_hedge_mamba import HedgeMambaMixerMLX
# Build a minimal MLX mixer just for the scan
mlx_mixer_bench = HedgeMambaMixerMLX(dim=384, num_heads=6, state_size=64)
# Use the raw scan function from mlx_hedge_mamba
_A_mx  = -mx.exp(mlx_mixer_bench.A_log.astype(mx.float32))
_Adup  = mx.concatenate([_A_mx, _A_mx], axis=0)

def mlx_scan_bench():
    # Mirrors what HedgeMambaMixerMLX._selective_scan does
    h = h0_mx
    outputs = []
    for t in range(L):
        dA  = mx.exp(mx.expand_dims(dt_mx[:, t], -1) * mx.expand_dims(_Adup, 0))
        dBu = mx.expand_dims(dt_mx[:, t] * u_mx[:, t], -1) * mx.expand_dims(Bf_mx[:, t], 1)
        h   = dA * h + dBu
        outputs.append((h * mx.expand_dims(Cf_mx[:, t], 1)).sum(axis=-1))
    y = mx.stack(outputs, axis=1)
    mx.eval(y)
    return y

# Warmup
for _ in range(5):
    mlx_scan_bench()

t0 = time.perf_counter()
for _ in range(N_SCAN):
    mlx_scan_bench()
lat_mlx = (time.perf_counter()-t0)/N_SCAN*1000

print(f"""
┌─────────────────────────────────────────────────────────────────┐
│  Training scan only  (B=2, L=448, D2=768, Ns=128, {N_SCAN} runs)     │
├──────────────────────────────────────────┬──────────────────────┤
│  Implementation                          │  Time per scan (ms)  │
├──────────────────────────────────────────┼──────────────────────┤
│  [B] PyTorch loop  (448 Python iters)    │          {lat_py:>8.1f}    │
│  [C] Metal kernel  (1 dispatch, MLX)     │          {lat_metal:>8.1f}    │
│  [MLX] MLX lazy-eval  (fused by MLX)     │          {lat_mlx:>8.1f}    │
├──────────────────────────────────────────┼──────────────────────┤
│  Metal kernel speedup vs Python loop     │          {lat_py/lat_metal:>8.1f}×    │
│  MLX lazy-eval speedup vs Python loop    │          {lat_py/lat_mlx:>8.1f}×    │
│  Metal kernel vs MLX lazy-eval           │          {lat_mlx/lat_metal:>8.1f}×    │
└──────────────────────────────────────────┴──────────────────────┘

Key insight
───────────
  Inference uses h_cache (L=1/step) → Metal kernel bypassed.
    Student still faster than teacher because O(1) state update
    beats O(seq_len) KV-cache grow in standard Whisper.
  Training uses full-sequence scan (L≈448) → Metal kernel fires,
    giving {lat_py/lat_metal:.1f}× speedup over the Python loop.
""")
