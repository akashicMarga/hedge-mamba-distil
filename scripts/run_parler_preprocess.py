#!/usr/bin/env python -u
"""Offline preprocessing: DAC-encode audio and apply delay pattern.

Run once before Stage 1/2 training to build the .npz cache.
Uses IndicParlerTTS from mlx-audio-train — no parler_tts pip package needed.

Usage::

    # Train split (full)
    python scripts/run_parler_preprocess.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --split train \
        --out_dir ./data/parler_distil

    # Validation split
    python scripts/run_parler_preprocess.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --split validation \
        --out_dir ./data/parler_distil

    # Debug: 100 samples only (~2 min)
    python scripts/run_parler_preprocess.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --debug --out_dir ./data/parler_distil_debug

Outputs one .npz per sample under {out_dir}/{split}/.
Each file: description_ids, attention_mask, prompt_ids, audio_tokens (T,9), labels (T,9).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlx_audio_train", required=True,
        help="Path to mlx-audio-train repo (IndicParlerTTS + MLX DAC)",
    )
    parser.add_argument(
        "--dataset", default="ai4b-hf/GLOBE-annotated",
        help="HuggingFace dataset id. Default: ai4b-hf/GLOBE-annotated "
             "(the exact data indic-parler-tts was trained on, 567K samples, 90 GB audio)",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split: train / validation / test",
    )
    parser.add_argument(
        "--out_dir", default="./data/parler_distil",
        help="Root output directory; files go into {out_dir}/{split}/",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of samples",
    )
    parser.add_argument(
        "--max_audio_len_s", type=float, default=6.0,
        help="Truncate audio longer than this (default 6 s ≈ 512 frames at 44100/512)",
    )
    parser.add_argument(
        "--lang_filter", default=None,
        help="Keep only rows with this language (e.g. Hindi, Tamil, Telugu). "
             "None = all languages. GLOBE-annotated language values are full "
             "English names: 'Hindi', 'Tamil', 'Telugu', 'Kannada', etc.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Quick smoke-test: process 100 samples only",
    )
    args = parser.parse_args()

    if args.debug:
        args.max_samples = args.max_samples or 100
        print("[preprocess] --debug: capping at 100 samples", flush=True)

    sys.path.insert(0, args.mlx_audio_train)

    from src.mlx.parler_data import preprocess_and_cache

    preprocess_and_cache(
        hf_dataset_id=args.dataset,
        out_dir=args.out_dir,
        split=args.split,
        max_samples=args.max_samples,
        max_audio_len_s=args.max_audio_len_s,
        lang_filter=args.lang_filter,
    )


if __name__ == "__main__":
    main()
