#!/usr/bin/env python
"""Live microphone demo for WhisperMamba (Fix A: use_cache=False).

Usage:
    python scripts/mic_demo.py                  # MPS (Mac)
    python scripts/mic_demo.py --device cuda    # NVIDIA GPU
    python scripts/mic_demo.py --device cpu     # CPU fallback

Press ENTER to start recording, ENTER again to stop and transcribe.
Ctrl-C to quit.

Dependencies (if missing):
    pip install sounddevice numpy scipy
"""
import sys
import warnings
import logging
import argparse
import threading
import numpy as np
from pathlib import Path

# Suppress noisy HuggingFace generation warnings
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Parse args ────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--device", default="mps", help="cpu | mps | cuda")
parser.add_argument("--ckpt", default="checkpoints/whisper_mamba/whisper_mamba_final.pt")
parser.add_argument("--model_id", default="openai/whisper-tiny")
parser.add_argument("--state_size", type=int, default=32)
args = parser.parse_args()

# ── Imports ───────────────────────────────────────────────────────────────────
try:
    import sounddevice as sd
except ImportError:
    print("sounddevice not installed. Run:  pip install sounddevice")
    sys.exit(1)

import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.student.whisper_mamba import WhisperMambaStudent

SAMPLE_RATE = 16_000   # Whisper expects 16 kHz mono

# ── Load model ────────────────────────────────────────────────────────────────
print("Loading WhisperMamba...", flush=True)
teacher   = WhisperForConditionalGeneration.from_pretrained(args.model_id, torch_dtype=torch.float32)
student   = WhisperMambaStudent(teacher, state_size=args.state_size)
student.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location="cpu"))
student   = student.to(args.device)
student.eval()
processor = WhisperProcessor.from_pretrained(args.model_id)

print(f"Model ready on {args.device}  |  checkpoint: {args.ckpt}", flush=True)
print(f"Fix B active: SSM h-cache + conv-cache (O(L) per decode step)\n", flush=True)


# ── Audio helpers ─────────────────────────────────────────────────────────────

def record_until_enter() -> np.ndarray:
    """Record from the default mic until the user presses ENTER."""
    chunks = []
    recording = threading.Event()
    recording.set()

    def _callback(indata, frames, time, status):
        if status:
            print(f"  [audio warning] {status}", flush=True)
        if recording.is_set():
            chunks.append(indata[:, 0].copy())   # mono

    stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                             dtype="float32", callback=_callback)
    with stream:
        input()           # blocks until ENTER
        recording.clear()

    return np.concatenate(chunks) if chunks else np.zeros(SAMPLE_RATE, dtype=np.float32)


def transcribe(audio: np.ndarray) -> str:
    """Run WhisperMamba on a numpy float32 array at 16 kHz."""
    # Whisper feature extraction handles normalisation
    inputs = processor(audio, sampling_rate=SAMPLE_RATE,
                       return_tensors="pt", return_attention_mask=True)
    features = inputs.input_features.to(args.device)        # (1, 80, 3000)
    attn_mask = inputs.attention_mask.to(args.device)       # silence the warning

    with torch.no_grad():
        ids = student.generate(
            features,
            attention_mask=attn_mask,
            language="en",
            task="transcribe",
            repetition_penalty=1.1,
        )
    return processor.decode(ids[0], skip_special_tokens=True)


# ── Main loop ─────────────────────────────────────────────────────────────────

print("=" * 55)
print("  WhisperMamba — Live Microphone Demo")
print("=" * 55)
print("Press ENTER to start recording.")
print("Press ENTER again to stop and transcribe.")
print("Ctrl-C to quit.\n")

try:
    while True:
        input("▶  [ENTER to record] ")
        print("🎙  Recording... (press ENTER to stop)", flush=True)

        audio = record_until_enter()
        duration = len(audio) / SAMPLE_RATE
        print(f"   Captured {duration:.1f}s — transcribing...", flush=True)

        text = transcribe(audio)

        print(f"\n📝  Transcription:\n    {text}\n")
        print("-" * 55)

except KeyboardInterrupt:
    print("\nBye!")
