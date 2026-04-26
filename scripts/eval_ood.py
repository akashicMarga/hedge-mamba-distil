#!/usr/bin/env python -u
"""Out-of-distribution WER evaluation — teacher vs student.

Evaluates on three datasets the student never saw during training:

  librispeech_test.clean  — in-distribution, standard benchmark
  librispeech_test.other  — harder speech (noisier, more diverse speakers)
  voxpopuli_en            — truly OOD: European Parliament recordings,
                            spontaneous speech, many accents

Usage:
    python scripts/eval_ood.py
    python scripts/eval_ood.py --device cuda --max_samples 500
    python scripts/eval_ood.py --out results/ood_wer.json
"""
import argparse, json, sys, warnings, io
warnings.filterwarnings("ignore")
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch, jiwer, numpy as np, soundfile as sf
from datasets import load_dataset, Audio
from torch.utils.data import DataLoader
from transformers import WhisperForConditionalGeneration, WhisperProcessor

from src.student.whisper_mamba import WhisperMambaStudent

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ckpt",        default="checkpoints/whisper_mamba/whisper_mamba_final.pt")
parser.add_argument("--hf_model",    default="openai/whisper-tiny")
parser.add_argument("--device",      default="mps")
parser.add_argument("--state_size",  type=int, default=64)
parser.add_argument("--batch_size",  type=int, default=8)
parser.add_argument("--max_samples", type=int, default=None,
                    help="Cap per dataset (None = full set)")
parser.add_argument("--out",         default="results/ood_wer.json")
args = parser.parse_args()

DEVICE = args.device

# ── Dataset definitions ───────────────────────────────────────────────────────

def _load_librispeech(config):
    ds = load_dataset("librispeech_asr", config, split="test")
    ds = ds.cast_column("audio", Audio(decode=False))
    return ds, "audio", "text"

DATASETS = {
    "librispeech_test.clean": lambda: _load_librispeech("clean"),
    "librispeech_test.other": lambda: _load_librispeech("other"),
}

# ── Audio → mel helper ────────────────────────────────────────────────────────

def _to_mel(example, audio_col, processor):
    """Extract numpy audio array from a dataset example (handles both
    HF decoded-audio dicts and raw-bytes dicts)."""
    info = example[audio_col]

    if isinstance(info, dict):
        if "array" in info:
            # FLEURS / decoded Audio
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

    feats = processor.feature_extractor(
        arr, sampling_rate=sr, return_tensors="np"
    ).input_features[0]   # (80, 3000)
    return feats


def _normalise(text: str) -> str:
    """Lowercase + strip punctuation for fair WER comparison.

    The student is trained on lowercased LibriSpeech text; the teacher
    outputs mixed-case with punctuation.  Normalising both ensures WER
    reflects word accuracy, not capitalisation differences.
    """
    import re
    text = text.lower()
    text = re.sub(r"[^\w\s']", "", text)   # keep apostrophes, drop other punct
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ── Evaluate one model on one dataset ────────────────────────────────────────

@torch.no_grad()
def evaluate(model, processor, ds, audio_col, text_col,
             batch_size=8, max_samples=None, desc=""):
    """Returns (wer_float, list_of_refs, list_of_preds)."""
    n = len(ds) if max_samples is None else min(max_samples, len(ds))
    ds = ds.select(range(n))

    refs, preds = [], []
    buf_feats, buf_refs = [], []

    def _flush():
        if not buf_feats:
            return
        feats = torch.tensor(np.stack(buf_feats)).to(DEVICE)  # (B, 80, 3000)
        ids = model.generate(
            feats, language="en", task="transcribe",
            max_new_tokens=128, repetition_penalty=1.1,
        )
        for i_id, ref in zip(ids, buf_refs):
            pred = processor.decode(i_id, skip_special_tokens=True)
            preds.append(_normalise(pred))
            refs.append(_normalise(ref))
        buf_feats.clear()
        buf_refs.clear()

    for i, ex in enumerate(ds):
        if i % 50 == 0:
            print(f"  {desc}  [{i}/{n}]", flush=True)
        try:
            feats = _to_mel(ex, audio_col, processor)
        except Exception as e:
            print(f"  skip sample {i}: {e}")
            continue

        buf_feats.append(feats)
        buf_refs.append(str(ex[text_col]))

        if len(buf_feats) >= batch_size:
            _flush()

    _flush()

    wer = jiwer.wer(refs, preds) * 100 if refs else float("nan")
    return wer, refs, preds


# ── Load models ───────────────────────────────────────────────────────────────

print("Loading teacher...", flush=True)
teacher = WhisperForConditionalGeneration.from_pretrained(
    args.hf_model, torch_dtype=torch.float32
).to(DEVICE).eval()
processor = WhisperProcessor.from_pretrained(args.hf_model)

print("Building student...", flush=True)
student = WhisperMambaStudent(teacher, state_size=args.state_size).to(DEVICE)
student.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location="cpu"))
student.eval()

# ── Run evaluation ────────────────────────────────────────────────────────────

results = {}

for ds_name, loader_fn in DATASETS.items():
    print(f"\n{'='*60}", flush=True)
    print(f"  Dataset: {ds_name}", flush=True)
    print(f"{'='*60}", flush=True)

    try:
        ds, audio_col, text_col = loader_fn()
        n_total = len(ds)
        n_eval  = args.max_samples or n_total
        print(f"  Total samples: {n_total:,}  →  evaluating {min(n_eval, n_total):,}", flush=True)
    except Exception as e:
        print(f"  Failed to load {ds_name}: {e}", flush=True)
        results[ds_name] = {"error": str(e)}
        continue

    print("  Running teacher...", flush=True)
    t_wer, t_refs, t_preds = evaluate(
        teacher, processor, ds, audio_col, text_col,
        batch_size=args.batch_size, max_samples=args.max_samples,
        desc=f"[teacher/{ds_name}]",
    )

    print("  Running student...", flush=True)
    s_wer, s_refs, s_preds = evaluate(
        student, processor, ds, audio_col, text_col,
        batch_size=args.batch_size, max_samples=args.max_samples,
        desc=f"[student/{ds_name}]",
    )

    results[ds_name] = {
        "n_samples":  len(t_refs),
        "teacher_wer": round(t_wer, 2),
        "student_wer": round(s_wer, 2),
        "gap":         round(s_wer - t_wer, 2),
        "samples": [
            {"ref": r, "teacher": tp, "student": sp}
            for r, tp, sp in zip(t_refs[:10], t_preds[:10], s_preds[:10])
        ],
    }

    print(f"\n  Teacher WER: {t_wer:.1f}%")
    print(f"  Student WER: {s_wer:.1f}%")
    print(f"  Gap:         {s_wer - t_wer:+.1f}%")

# ── Print summary table ───────────────────────────────────────────────────────

print(f"\n{'='*60}")
print(f"  {'Dataset':<30}  {'Teacher':>8}  {'Student':>8}  {'Gap':>6}")
print(f"  {'-'*56}")
for ds_name, r in results.items():
    if "error" in r:
        print(f"  {ds_name:<30}  {'ERROR':>8}")
    else:
        print(f"  {ds_name:<30}  {r['teacher_wer']:>7.1f}%  {r['student_wer']:>7.1f}%  {r['gap']:>+6.1f}%")
print(f"{'='*60}")

# ── Save ──────────────────────────────────────────────────────────────────────

Path(args.out).parent.mkdir(parents=True, exist_ok=True)
with open(args.out, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved → {args.out}")
