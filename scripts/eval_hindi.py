#!/usr/bin/env python -u
"""Hindi cross-lingual evaluation — teacher vs student on Kathbath.

The student was trained entirely on English LibriSpeech.
The teacher (openai/whisper-tiny) is multilingual (99 languages incl. Hindi).

This measures how much the English fine-tuning damaged the student's
ability to handle a language it was never trained on.

Teacher: forced language=hi, task=transcribe
Student: two modes —
  (a) language=hi — forces Hindi output even though it was trained on English
  (b) language=en — lets it respond in English (expected garbage on Hindi audio)

We report WER for both models in Hindi-forced mode.
WER is computed on Devanagari text; jiwer handles Unicode natively.

Usage:
    python scripts/eval_hindi.py
    python scripts/eval_hindi.py --n_samples 100 --out results/hindi_wer.json
"""
import argparse, json, re, sys, warnings, io
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, jiwer, numpy as np, soundfile as sf
from datasets import load_dataset
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.student.whisper_mamba import WhisperMambaStudent

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",       default="checkpoints/whisper_mamba/whisper_mamba_final.pt")
parser.add_argument("--hf_model",   default="openai/whisper-tiny")
parser.add_argument("--device",     default="mps")
parser.add_argument("--state_size", type=int, default=64)
parser.add_argument("--n_samples",  type=int, default=100)
parser.add_argument("--out",        default="results/hindi_wer.json")
args = parser.parse_args()

DEVICE = args.device


# ── Load data ─────────────────────────────────────────────────────────────────

print("Loading Kathbath Hindi test set...", flush=True)
ds = load_dataset("ai4bharat/Kathbath", "hindi", split="test")
ds = ds.select(range(min(args.n_samples, len(ds))))
print(f"Loaded {len(ds)} samples. Columns: {ds.column_names}", flush=True)

# Discover text column
text_col = next(
    c for c in ds.column_names
    if any(k in c.lower() for k in ["transcript", "text", "sentence", "normalized"])
)
print(f"Using text column: {text_col!r}", flush=True)


def _to_mel(example, processor):
    info = example["audio"]
    if isinstance(info, dict):
        if "array" in info:
            arr, sr = info["array"], info["sampling_rate"]
        elif info.get("bytes"):
            arr, sr = sf.read(io.BytesIO(info["bytes"]))
        else:
            arr, sr = sf.read(info["path"])
    else:
        raise ValueError(f"Unknown audio format: {type(info)}")
    arr = np.array(arr, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr.mean(axis=1)
    return processor.feature_extractor(
        arr, sampling_rate=sr, return_tensors="np"
    ).input_features[0]   # (80, 3000)


def _norm_hindi(text: str) -> str:
    """Minimal normalisation for Devanagari: strip punctuation, collapse spaces."""
    text = re.sub(r"[।॥,\.!?\"'();:\-]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ── Load models ───────────────────────────────────────────────────────────────

print("Loading teacher...", flush=True)
# Use Hindi processor for teacher to get correct forced_decoder_ids
teacher_proc_hi = WhisperProcessor.from_pretrained(args.hf_model, language="hi", task="transcribe")
teacher = WhisperForConditionalGeneration.from_pretrained(
    args.hf_model, torch_dtype=torch.float32
).to(DEVICE).eval()

print("Building student...", flush=True)
student = WhisperMambaStudent(teacher, state_size=args.state_size).to(DEVICE)
student.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location="cpu"))
student.eval()

print(f"\nEvaluating {len(ds)} Hindi samples...\n", flush=True)

# ── Evaluate ──────────────────────────────────────────────────────────────────

refs, t_preds, s_preds_hi, s_preds_en = [], [], [], []

for i, ex in enumerate(ds):
    if i % 20 == 0:
        print(f"  [{i}/{len(ds)}]", flush=True)

    try:
        feats_np = _to_mel(ex, teacher_proc_hi)
    except Exception as e:
        print(f"  skip {i}: {e}")
        continue

    feats = torch.tensor(feats_np).unsqueeze(0).to(DEVICE)
    ref   = _norm_hindi(str(ex[text_col]))

    with torch.no_grad():
        # Teacher — Hindi forced
        t_ids = teacher.generate(
            feats,
            language="hi", task="transcribe",
            max_new_tokens=128, repetition_penalty=1.1,
        )
        t_text = teacher_proc_hi.decode(t_ids[0], skip_special_tokens=True)

        # Student — Hindi forced (tests multilingual retention)
        s_ids_hi = student.generate(
            feats,
            language="hi", task="transcribe",
            max_new_tokens=128, repetition_penalty=1.1,
        )
        s_text_hi = teacher_proc_hi.decode(s_ids_hi[0], skip_special_tokens=True)

        # Student — English forced (shows what it defaults to)
        s_ids_en = student.generate(
            feats,
            language="en", task="transcribe",
            max_new_tokens=128, repetition_penalty=1.1,
        )
        s_text_en = teacher_proc_hi.decode(s_ids_en[0], skip_special_tokens=True)

    refs.append(ref)
    t_preds.append(_norm_hindi(t_text))
    s_preds_hi.append(_norm_hindi(s_text_hi))
    s_preds_en.append(s_text_en)  # English output — no Devanagari norm needed

# ── WER ───────────────────────────────────────────────────────────────────────

t_wer      = jiwer.wer(refs, t_preds)      * 100
s_wer_hi   = jiwer.wer(refs, s_preds_hi)   * 100
# For English-forced student: WER vs Hindi reference is not meaningful,
# but we report CER to quantify how wrong the output is
s_cer_en   = jiwer.cer(refs, s_preds_en)   * 100

print(f"\n{'='*55}")
print(f"  Hindi evaluation — Kathbath test ({len(refs)} samples)")
print(f"{'='*55}")
print(f"  Teacher (Whisper-tiny multilingual, lang=hi):  {t_wer:.1f}% WER")
print(f"  Student (fine-tuned English, lang=hi forced):  {s_wer_hi:.1f}% WER")
print(f"  Student (lang=en):  {s_cer_en:.1f}% CER vs Hindi ref  (English output)")
print(f"{'='*55}\n")

# Sample outputs
print("Sample predictions (first 5):\n")
print(f"{'REF':<45}  {'TEACHER':^40}  {'STUDENT (hi)':^40}")
print("-" * 130)
for r, t, s in list(zip(refs, t_preds, s_preds_hi))[:5]:
    print(f"{r[:43]:<45}  {t[:38]:^40}  {s[:38]:^40}")

# ── Save ──────────────────────────────────────────────────────────────────────

result = {
    "dataset": "ai4bharat/Kathbath (hindi, test)",
    "n_samples": len(refs),
    "teacher_wer_hi": round(t_wer, 2),
    "student_wer_hi_forced": round(s_wer_hi, 2),
    "student_cer_en_forced": round(s_cer_en, 2),
    "note": (
        "Teacher is openai/whisper-tiny (multilingual). "
        "Student was fine-tuned on English LibriSpeech only — "
        "Hindi performance shows cross-lingual retention after English fine-tuning."
    ),
    "samples": [
        {"ref": r, "teacher": t, "student_hi": s, "student_en": e}
        for r, t, s, e in list(zip(refs, t_preds, s_preds_hi, s_preds_en))[:10]
    ],
}

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
print(f"Results saved → {args.out}")
