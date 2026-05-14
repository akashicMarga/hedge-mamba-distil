#!/usr/bin/env python -u
"""Offline preprocessing: DAC-encode audio and apply delay pattern.

Run once before Stage 1/2 training to build the .npz cache.

Usage::

    # Train split
    python scripts/run_parler_preprocess.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --split train \
        --out_dir ./data/parler_distil

    # Validation split
    python scripts/run_parler_preprocess.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --split validation \
        --out_dir ./data/parler_distil

Outputs one .npz file per sample under {out_dir}/{split}/.
Each file contains: description_ids, attention_mask, prompt_ids,
audio_tokens (T,9), labels (T,9).
"""
import argparse
import sys
from pathlib import Path

# ── make repo importable ──────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mlx_audio_train", required=True,
        help="Path to mlx-audio-train repo (needed for IndicParlerTTS + DAC)",
    )
    parser.add_argument(
        "--dataset", default="parler-tts/parler-tts-mini-v1",
        help="HuggingFace dataset id with (description, text, audio) columns",
    )
    parser.add_argument(
        "--split", default="train",
        help="Dataset split to process (train / validation / test)",
    )
    parser.add_argument(
        "--out_dir", default="./data/parler_distil",
        help="Root output directory; files go into {out_dir}/{split}/",
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Limit number of samples (useful for debugging)",
    )
    parser.add_argument(
        "--max_audio_len_s", type=float, default=6.0,
        help="Truncate audio longer than this many seconds (default 6s ≈ 512 frames)",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="Torch device for DAC encoder: cpu | mps | cuda",
    )
    args = parser.parse_args()

    # Add mlx-audio-train to path (provides IndicParlerTTS + MLX DAC)
    sys.path.insert(0, args.mlx_audio_train)

    from src.mlx.parler_data import preprocess_and_cache

    preprocess_and_cache(
        hf_dataset_id=args.dataset,
        out_dir=args.out_dir,
        split=args.split,
        max_samples=args.max_samples,
        max_audio_len_s=args.max_audio_len_s,
        device=args.device,
    )


if __name__ == "__main__":
    main()
