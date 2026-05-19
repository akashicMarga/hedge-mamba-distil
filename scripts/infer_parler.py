#!/usr/bin/env python -u
"""Inference: generate speech from text using the distilled student model.

Usage (with p11 env):
    python scripts/infer_parler.py \
        --mlx_audio_train /Users/akashsingh/Documents/exps/mlx-audio-train \
        --stage2_ckpt ./checkpoints/parler_mamba_en/stage2_epoch_1 \
        --state_size 128 \
        --description "A clear male voice speaking English with good pronunciation." \
        --text "The quick brown fox jumps over the lazy dog." \
        --out student_out.wav

    # Compare with teacher:
    python scripts/infer_parler.py ... --also_teacher

    # Play immediately (macOS):
    python scripts/infer_parler.py ... --play
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx_audio_train", required=True)
    parser.add_argument("--stage2_ckpt",     required=True,
                        help="Path prefix without .npz/.json")
    parser.add_argument("--state_size",      type=int, default=128)
    parser.add_argument("--description", default=(
        "A clear male voice speaks English at a moderate pace "
        "with high audio quality and no background noise."
    ))
    parser.add_argument("--text", default=(
        "Hello! This is the HedgeMamba distilled speech synthesis model."
    ))
    parser.add_argument("--out",             default="student_out.wav",
                        help="Output .wav path for student")
    parser.add_argument("--also_teacher",    action="store_true",
                        help="Also generate and save teacher output for comparison")
    parser.add_argument("--teacher_out",     default="teacher_out.wav")
    parser.add_argument("--max_audio_len_s", type=float, default=10.0)
    parser.add_argument("--temperature",     type=float, default=1.0)
    parser.add_argument("--top_k",          type=int,   default=50)
    parser.add_argument("--play",            action="store_true",
                        help="Play output via afplay (macOS) after saving")
    args = parser.parse_args()

    sys.path.insert(0, args.mlx_audio_train)

    import numpy as np
    import soundfile as sf
    from models.indic_parler_tts.model import IndicParlerTTS
    from models.indic_parler_tts.generate import load_model, generate
    from src.mlx.parler_model import ParlerMambaMLX
    from src.mlx import parler_checkpoint as ckpt_io

    # ── Load teacher ──────────────────────────────────────────────────────────
    print("Loading teacher model ...", flush=True)
    teacher, tokenizers = load_model()

    # ── Load student ──────────────────────────────────────────────────────────
    print(f"Loading student model (state_size={args.state_size}) ...", flush=True)
    student_base = IndicParlerTTS.from_pretrained()
    student = ParlerMambaMLX(student_base, state_size=args.state_size)
    ckpt_io.load(student, args.stage2_ckpt)
    student.freeze()
    # Re-attach DAC decoder — it was popped during init to avoid breaking
    # model.update() in training, but generate() needs it for decoding.
    if student._dac is not None and not hasattr(student.model, 'dac'):
        student.model.dac = student._dac
    print(student.param_summary(), flush=True)

    print(f"\nDescription : {args.description}")
    print(f"Text        : {args.text}\n")

    # ── Student generation ────────────────────────────────────────────────────
    print("Generating (student) ...", flush=True)
    t0 = time.perf_counter()
    student_wav = generate(
        student.model, tokenizers,
        description=args.description,
        text=args.text,
        max_audio_length_s=args.max_audio_len_s,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    student_time = time.perf_counter() - t0
    audio_dur = len(student_wav) / 44100
    rtf = student_time / max(audio_dur, 1e-6)

    sf.write(args.out, student_wav, 44100)
    print(f"Student : {audio_dur:.1f}s audio in {student_time:.1f}s  (RTF={rtf:.3f})")
    print(f"Saved   : {args.out}")

    # ── Teacher generation (optional) ─────────────────────────────────────────
    if args.also_teacher:
        print("\nGenerating (teacher) ...", flush=True)
        t0 = time.perf_counter()
        teacher_wav = generate(
            teacher, tokenizers,
            description=args.description,
            text=args.text,
            max_audio_length_s=args.max_audio_len_s,
            temperature=args.temperature,
            top_k=args.top_k,
        )
        teacher_time = time.perf_counter() - t0
        teacher_dur = len(teacher_wav) / 44100
        sf.write(args.teacher_out, teacher_wav, 44100)
        print(f"Teacher : {teacher_dur:.1f}s audio in {teacher_time:.1f}s  "
              f"(RTF={teacher_time/max(teacher_dur,1e-6):.3f})")
        print(f"Saved   : {args.teacher_out}")

        try:
            import librosa
            from scripts.eval_parler import compute_mcd
            mcd = compute_mcd(teacher_wav, student_wav, sr=44100)
            print(f"MCD     : {mcd:.2f} dB  (lower = closer to teacher)")
        except Exception:
            pass

    # ── Play ──────────────────────────────────────────────────────────────────
    if args.play:
        import subprocess
        print(f"\nPlaying student output ...", flush=True)
        subprocess.run(["afplay", args.out])
        if args.also_teacher:
            print("Playing teacher output ...")
            subprocess.run(["afplay", args.teacher_out])


if __name__ == "__main__":
    main()
