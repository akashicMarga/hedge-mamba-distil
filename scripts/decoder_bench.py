#!/usr/bin/env python
"""Decoder-only speed benchmark + WER comparison.

Encoder outputs are pre-computed once and reused, so timing reflects
purely the decoder autoregressive generation loop.
"""
import sys, time, warnings, logging
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
sys.path.insert(0, ".")

import torch
import jiwer
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from src.student.whisper_mamba import WhisperMambaStudent
from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator
from torch.utils.data import Subset, DataLoader

DEVICE      = "mps"
CKPT        = "checkpoints/whisper_mamba/whisper_mamba_final.pt"
MODEL_ID    = "openai/whisper-tiny"
N_SAMPLES   = 30
WARMUP      = 3
GEN_KWARGS  = dict(language="en", task="transcribe",
                   max_new_tokens=128, repetition_penalty=1.1)

# ── Load models ───────────────────────────────────────────────────────────────
print("Loading...", flush=True)
teacher_model = WhisperForConditionalGeneration.from_pretrained(
    MODEL_ID, torch_dtype=torch.float32).to(DEVICE).eval()

student_model = WhisperMambaStudent(
    WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32),
    state_size=64)
student_model.load_state_dict(torch.load(CKPT, weights_only=True, map_location="cpu"))
student_model = student_model.to(DEVICE).eval()

processor = WhisperProcessor.from_pretrained(MODEL_ID)

# ── Data ──────────────────────────────────────────────────────────────────────
_, val_loader, _ = make_librispeech_loaders(
    model_id=MODEL_ID, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=1, max_label_length=448,
)
loader = DataLoader(
    Subset(val_loader.dataset, range(WARMUP + N_SAMPLES)),
    batch_size=1, shuffle=False, collate_fn=LibriSpeechCollator(), num_workers=0,
)
batches = list(loader)

# ── Pre-compute encoder outputs (shared, identical for both models) ────────────
print("Pre-computing encoder outputs...", flush=True)
encoder_outputs_list = []
refs = []
with torch.no_grad():
    for batch in batches:
        feats = batch["input_features"].to(DEVICE)
        enc_out = teacher_model.model.encoder(feats)   # same encoder in both
        encoder_outputs_list.append(enc_out)
        refs.append(processor.decode(batch["labels"][0].clamp(min=0), skip_special_tokens=True))


def sync():
    if DEVICE == "mps":  torch.mps.synchronize()
    elif DEVICE == "cuda": torch.cuda.synchronize()


def bench_decoder(model, name):
    """Time decoder-only generation using pre-computed encoder outputs."""
    latencies, n_tokens_list, preds = [], [], []

    for i, (enc_out, batch) in enumerate(zip(encoder_outputs_list, batches)):
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            ids = model.generate(
                batch["input_features"].to(DEVICE),   # needed for Whisper's generate()
                encoder_outputs=enc_out,              # skip encoder — use precomputed
                **GEN_KWARGS,
            )
        sync()
        elapsed = (time.perf_counter() - t0) * 1000

        n_tok = ids.shape[-1]
        pred  = processor.decode(ids[0], skip_special_tokens=True)

        if i >= WARMUP:
            latencies.append(elapsed)
            n_tokens_list.append(n_tok)
            preds.append(pred)

    mean_lat = sum(latencies) / len(latencies)
    tok_s    = sum(n_tokens_list) / (sum(latencies) / 1000)
    wer      = jiwer.wer(refs[WARMUP:], preds) * 100
    return mean_lat, tok_s, wer, preds


# ── Run ───────────────────────────────────────────────────────────────────────
print(f"\nBenchmarking decoder only  ({N_SAMPLES} samples, {WARMUP} warmup, device={DEVICE})\n")

print("── Teacher decoder (multi-head self-attention + KV cache) ──")
t_lat, t_tps, t_wer, t_preds = bench_decoder(teacher_model, "Teacher")

print("── Student decoder (SelectiveSSM + h/conv cache, Fix B)   ──")
s_lat, s_tps, s_wer, s_preds = bench_decoder(student_model, "Student")

speedup = t_lat / s_lat

# ── Results ───────────────────────────────────────────────────────────────────
print(f"\n{'═'*52}")
print(f"  Decoder-only results  (device={DEVICE})")
print(f"{'═'*52}")
print(f"  {'Metric':<28} {'Teacher':>10}  {'Student':>10}")
print(f"  {'-'*50}")
print(f"  {'Mean latency / utterance':<28} {t_lat:>9.0f}ms  {s_lat:>9.0f}ms")
print(f"  {'Tokens / second':<28} {t_tps:>10.1f}  {s_tps:>10.1f}")
print(f"  {'Decoder speedup':<28} {'—':>10}  {speedup:>9.2f}×")
print(f"  {'-'*50}")
print(f"  {'WER ↓':<28} {t_wer:>9.1f}%  {s_wer:>9.1f}%")
print(f"  {'WER gap (Student − Teacher)':<28} {s_wer - t_wer:>+10.1f}%")
print(f"{'═'*52}")

# ── Sample transcriptions (side by side) ─────────────────────────────────────
print(f"\n── Sample transcriptions (first 4) ────────────────────────")
for i in range(min(4, len(refs) - WARMUP)):
    ref  = refs[WARMUP + i]
    tp   = t_preds[i]
    sp   = s_preds[i]
    print(f"\n  Ref    : {ref!r}")
    print(f"  Teacher: {tp!r}")
    print(f"  Student: {sp!r}")
