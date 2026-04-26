#!/usr/bin/env python
"""Quick WER eval with Fix A (use_cache=False) on the existing final checkpoint."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import WhisperForConditionalGeneration
from src.student.whisper_mamba import WhisperMambaStudent
from src.data.librispeech import make_librispeech_loaders
from src.distill.whisper_train import compute_wer

DEVICE = "mps"
CKPT   = "checkpoints/whisper_mamba/whisper_mamba_final.pt"
MODEL  = "openai/whisper-tiny"

print("Loading teacher skeleton...", flush=True)
teacher = WhisperForConditionalGeneration.from_pretrained(MODEL, torch_dtype=torch.float32)

print("Building student...", flush=True)
student = WhisperMambaStudent(teacher, state_size=32)
student.load_state_dict(torch.load(CKPT, weights_only=True, map_location="cpu"))
student = student.to(DEVICE)
student.eval()

print(f"Loaded checkpoint: {CKPT}", flush=True)
print(f"use_cache will be forced False via generate() patch", flush=True)

_, val_loader, processor = make_librispeech_loaders(
    model_id=MODEL, language="en", task="transcribe",
    train_split="train.100", val_split="validation",
    batch_size=4, max_label_length=448,
)

# Restrict to a small subset for speed
from torch.utils.data import Subset, DataLoader
from src.data.librispeech import LibriSpeechCollator
val_loader = DataLoader(
    Subset(val_loader.dataset, range(40)),
    batch_size=4, shuffle=False,
    collate_fn=LibriSpeechCollator(), num_workers=0,
)

print("\nComputing WER (Fix A: use_cache=False)...", flush=True)
wer = compute_wer(student, val_loader, processor, DEVICE, max_batches=10)
print(f"\n>>> WER (Fix A) = {wer*100:.1f}%", flush=True)

# Show sample predictions
print("\n=== Sample Transcriptions ===", flush=True)
batch = next(iter(val_loader))
feats = batch["input_features"][:3].to(DEVICE)
labels = batch["labels"][:3]

with torch.no_grad():
    ids = student.generate(feats, language="en", task="transcribe",
                           max_new_tokens=128, repetition_penalty=1.1)

for i in range(3):
    pred = processor.decode(ids[i], skip_special_tokens=True)
    ref  = processor.decode(labels[i].clamp(min=0), skip_special_tokens=True)
    print(f"  [{i+1}] Ref:  {ref!r}", flush=True)
    print(f"       Pred: {pred!r}", flush=True)
