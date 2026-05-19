#!/usr/bin/env python -u
"""Audio quality eval: teacher vs. student ParlerTTS-Mamba.

Generates N samples using both teacher and student, then reports:
  - Mel Cepstral Distortion (MCD): lower is better, 0 = identical
  - Real-Time Factor (RTF): wall-clock / audio duration

Usage::

    python scripts/eval_parler.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --stage2_ckpt ./checkpoints/parler_mamba/stage2_epoch_5 \
        --num_samples 20

Optional::

    --state_size 64
    --max_audio_len 512
    --out_dir ./eval_out    # saves teacher/student .wav pairs for listening
    --dataset parler-tts/parler-tts-mini-v1
    --split validation
"""
import argparse
import sys
import time
import math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# MCD (Mel Cepstral Distortion)
# ---------------------------------------------------------------------------

def compute_mcd(wav_ref, wav_hyp, sr: int = 44100, n_mfcc: int = 13) -> float:
    """MCD between two waveforms (numpy float32 arrays).

    Uses librosa MFCCs.  Returns mean MCD in dB.  Lower = more similar.
    """
    import numpy as np
    try:
        import librosa
    except ImportError:
        raise ImportError("pip install librosa  # needed for MCD eval")

    def _mfcc(wav):
        return librosa.feature.mfcc(y=wav, sr=sr, n_mfcc=n_mfcc).T  # (T, n_mfcc)

    ref = _mfcc(wav_ref)
    hyp = _mfcc(wav_hyp)

    # Align lengths by truncating to shorter
    L = min(ref.shape[0], hyp.shape[0])
    ref, hyp = ref[:L], hyp[:L]

    diff  = ref - hyp
    mcd   = (10.0 / math.log(10)) * math.sqrt(2.0) * float(
        (diff ** 2).sum(axis=1).mean() ** 0.5
    )
    return mcd


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx_audio_train", required=True)
    parser.add_argument("--stage2_ckpt", required=True,
                        help="Path prefix (without .npz/.json) for stage2 checkpoint")
    parser.add_argument("--state_size",   type=int, default=64)
    parser.add_argument("--num_samples",  type=int, default=20)
    parser.add_argument("--max_audio_len",type=int, default=512)
    parser.add_argument("--dataset", default="ai4b-hf/GLOBE-annotated")
    parser.add_argument("--split",   default="test")
    parser.add_argument("--description_col", default="auto")
    parser.add_argument("--out_dir", default=None,
                        help="If set, saves wav pairs for manual listening")
    args = parser.parse_args()

    sys.path.insert(0, args.mlx_audio_train)

    import numpy as np
    import mlx.core as mx

    from datasets import load_dataset
    from models.indic_parler_tts.model import IndicParlerTTS
    from src.mlx.parler_model import ParlerMambaMLX
    from src.mlx import parler_checkpoint as ckpt_io

    if args.out_dir:
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Load teacher ──────────────────────────────────────────────────────────
    print("[eval] Loading teacher ...", flush=True)
    teacher = IndicParlerTTS.from_pretrained()
    teacher.freeze()

    # ── Load student ──────────────────────────────────────────────────────────
    print("[eval] Building student ...", flush=True)
    student_base = IndicParlerTTS.from_pretrained()
    student = ParlerMambaMLX(student_base, state_size=args.state_size)
    ckpt_io.load(student, args.stage2_ckpt)
    student.freeze()
    print(student.param_summary(), flush=True)

    # ── Load a few validation samples ─────────────────────────────────────────
    print(f"[eval] Loading {args.dataset} / {args.split} ...", flush=True)
    ds = load_dataset(args.dataset, split=args.split,
                      verification_mode="no_checks")
    ds = ds.select(range(min(args.num_samples, len(ds))))

    # Auto-detect description column
    _DESC_CANDIDATES = ["text_description", "description", "caption", "prompt"]
    if args.description_col == "auto":
        first = ds[0]
        desc_col = next((c for c in _DESC_CANDIDATES if c in first), None)
        if desc_col is None:
            raise KeyError(f"No description column found. Tried {_DESC_CANDIDATES}. "
                           f"Available: {list(first.keys())}")
        print(f"[eval] Using description column: '{desc_col}'", flush=True)
    else:
        desc_col = args.description_col

    mcd_vals   = []
    rtf_vals   = []

    for idx, sample in enumerate(ds):
        description = sample[desc_col]
        text        = sample["text"]
        ref_wav     = np.array(sample["audio"]["array"], dtype=np.float32)
        ref_sr      = sample["audio"]["sampling_rate"]

        # ── Teacher generate ────────────────────────────────────────────────
        t0 = time.perf_counter()
        teacher_wav = teacher.generate(description=description, text=text)
        teacher_time = time.perf_counter() - t0
        teacher_wav = np.array(teacher_wav)

        # ── Student generate ────────────────────────────────────────────────
        t0 = time.perf_counter()
        student_wav = student.model.generate(description=description, text=text)
        student_time = time.perf_counter() - t0
        student_wav = np.array(student_wav)

        # ── Metrics ─────────────────────────────────────────────────────────
        try:
            mcd = compute_mcd(teacher_wav.squeeze(), student_wav.squeeze(), sr=44100)
        except Exception as e:
            print(f"  [eval] MCD failed for sample {idx}: {e}")
            mcd = float("nan")

        audio_dur = student_wav.shape[-1] / 44100
        rtf       = student_time / max(audio_dur, 1e-6)

        mcd_vals.append(mcd)
        rtf_vals.append(rtf)

        print(
            f"  [{idx+1:3d}/{len(ds)}] MCD={mcd:.2f} dB | "
            f"student_time={student_time:.2f}s | RTF={rtf:.3f}",
            flush=True,
        )

        if args.out_dir:
            try:
                import soundfile as sf
                sf.write(f"{args.out_dir}/{idx:04d}_teacher.wav", teacher_wav.squeeze(), 44100)
                sf.write(f"{args.out_dir}/{idx:04d}_student.wav", student_wav.squeeze(), 44100)
            except Exception as e:
                print(f"  [eval] wav save failed: {e}")

    # ── Summary ───────────────────────────────────────────────────────────────
    valid_mcd = [v for v in mcd_vals if math.isfinite(v)]
    valid_rtf = [v for v in rtf_vals if math.isfinite(v)]
    print("\n" + "="*55)
    print(f"[eval] Samples evaluated : {len(ds)}")
    print(f"[eval] Mean MCD          : {sum(valid_mcd)/max(len(valid_mcd),1):.2f} dB  (lower = closer to teacher)")
    print(f"[eval] Mean RTF          : {sum(valid_rtf)/max(len(valid_rtf),1):.3f}  (< 1.0 = faster than real-time)")
    print("="*55)


if __name__ == "__main__":
    main()
