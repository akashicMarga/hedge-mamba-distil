#!/usr/bin/env python
"""Speed benchmark: Whisper-tiny teacher vs WhisperMamba student.

Architecture reminder
─────────────────────
  Teacher  : Whisper-tiny  (encoder + 4 self-attn decoder layers)
  Student  : WhisperMamba  (same frozen encoder + 4 Mamba SSM decoder layers)

Only the decoder self-attention differs.
  • Teacher decoder: standard multi-head self-attention + KV cache  → O(L·d) per step
  • Student decoder: SelectiveSSM + h/conv cache (Fix B)            → O(d·N) per step

We measure:
  - Mean latency per utterance  (ms)
  - Tokens generated per second
  - Speedup ratio

Usage:
    python scripts/speed_bench.py --device mps
    python scripts/speed_bench.py --device cpu
    python scripts/speed_bench.py --device cuda
"""
import sys, time, argparse, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.student.whisper_mamba import WhisperMambaStudent
from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator
from torch.utils.data import Subset, DataLoader

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device",     default="mps")
parser.add_argument("--ckpt",       default="checkpoints/whisper_mamba/whisper_mamba_final.pt")
parser.add_argument("--model_id",   default="openai/whisper-tiny")
parser.add_argument("--state_size", type=int, default=32)
parser.add_argument("--n_samples",  type=int, default=20, help="Val samples to benchmark on")
parser.add_argument("--warmup",     type=int, default=3,  help="Warmup runs (excluded from timing)")
parser.add_argument("--max_new_tokens", type=int, default=128)
args = parser.parse_args()

DEVICE = args.device
GEN_KWARGS = dict(language="en", task="transcribe",
                  max_new_tokens=args.max_new_tokens, repetition_penalty=1.1)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading models...", flush=True)
teacher = WhisperForConditionalGeneration.from_pretrained(
    args.model_id, torch_dtype=torch.float32).to(DEVICE).eval()

whisper_teacher_raw = WhisperForConditionalGeneration.from_pretrained(
    args.model_id, torch_dtype=torch.float32)
student = WhisperMambaStudent(whisper_teacher_raw, state_size=args.state_size)
student.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location="cpu"))
student = student.to(DEVICE).eval()

processor = WhisperProcessor.from_pretrained(args.model_id)

# ── Architecture summary ──────────────────────────────────────────────────────
t_params  = sum(p.numel() for p in teacher.parameters()) / 1e6
s_total   = sum(p.numel() for p in student.parameters()) / 1e6
s_train   = sum(p.numel() for p in student.parameters() if p.requires_grad) / 1e6

print(f"\n{'='*55}")
print(f"  Architecture Comparison")
print(f"{'='*55}")
print(f"  Teacher (Whisper-tiny)")
print(f"    Encoder : Conv + 4 transformer layers  [frozen in student]")
print(f"    Decoder : 4 × multi-head self-attention + KV cache")
print(f"    Params  : {t_params:.1f}M total")
print(f"\n  Student (WhisperMamba)")
print(f"    Encoder : identical frozen Whisper encoder")
print(f"    Decoder : 4 × SelectiveSSM (h-cache + conv-cache, Fix B)")
print(f"    Params  : {s_total:.1f}M total  |  {s_train:.1f}M trainable")
print(f"\n  Decode step complexity")
print(f"    Teacher : O(L·d) per step  (attention over KV cache grows with L)")
print(f"    Student : O(d·N) per step  (SSM state N={args.state_size}, fixed)")
print(f"{'='*55}\n")

# ── Data ──────────────────────────────────────────────────────────────────────
_, val_loader, _ = make_librispeech_loaders(
    model_id=args.model_id, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=1, max_label_length=448,
)
loader = DataLoader(
    Subset(val_loader.dataset, range(args.warmup + args.n_samples)),
    batch_size=1, shuffle=False,
    collate_fn=LibriSpeechCollator(), num_workers=0,
)
batches = [b for b in loader]


def run_benchmark(model, name, batches, warmup):
    """Returns (mean_latency_ms, tokens_per_sec, latencies)."""
    latencies   = []
    total_tokens = 0

    for i, batch in enumerate(batches):
        feats = batch["input_features"].to(DEVICE)

        if DEVICE == "mps":
            torch.mps.synchronize()
        elif DEVICE == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(feats, **GEN_KWARGS)
        if DEVICE == "mps":
            torch.mps.synchronize()
        elif DEVICE == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        n_tokens = ids.shape[-1]
        elapsed  = (t1 - t0) * 1000   # ms

        if i >= warmup:                # skip warmup
            latencies.append(elapsed)
            total_tokens += n_tokens

        tag = "(warmup)" if i < warmup else f"[{i - warmup + 1:2d}/{len(batches)-warmup}]"
        print(f"  {name} {tag}  {elapsed:6.0f} ms  {n_tokens} tokens", flush=True)

    mean_lat  = sum(latencies) / len(latencies)
    tok_per_s = total_tokens / (sum(latencies) / 1000)
    return mean_lat, tok_per_s, latencies


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"Benchmarking on {args.n_samples} samples  (warmup={args.warmup})  device={DEVICE}\n")

print("── Teacher (Whisper-tiny) ──────────────────────────────")
t_lat, t_tps, t_lats = run_benchmark(teacher, "Teacher", batches, args.warmup)

print("\n── Student (WhisperMamba, Fix B) ───────────────────────")
s_lat, s_tps, s_lats = run_benchmark(student, "Student", batches, args.warmup)

# ── Summary ───────────────────────────────────────────────────────────────────
speedup = t_lat / s_lat
print(f"\n{'='*55}")
print(f"  Results  (device={DEVICE})")
print(f"{'='*55}")
print(f"  {'':30s} {'Teacher':>10}  {'Student':>10}")
print(f"  {'Mean latency / utterance':30s} {t_lat:>9.0f}ms  {s_lat:>9.0f}ms")
print(f"  {'Tokens / second':30s} {t_tps:>10.1f}  {s_tps:>10.1f}")
print(f"  {'Speedup (Student / Teacher)':30s} {'':10}  {speedup:>9.2f}×")
print(f"{'='*55}")

if speedup >= 1:
    print(f"\n  ✅ Student is {speedup:.2f}× faster than teacher.")
else:
    print(f"\n  ℹ️  Student is {1/speedup:.2f}× slower (expected for tiny L — "
          f"SSM advantage grows with sequence length).")

print(f"\nNote: both models share the identical frozen Whisper encoder.")
print(f"Speed difference reflects ONLY the decoder self-attention → SSM swap.")
