#!/usr/bin/env python -u
"""Stage 1: cosine distillation — SSM layers learn to mimic teacher hiddens.

Usage::

    python scripts/run_parler_stage1.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --train_cache ./data/parler_distil/train \
        --val_cache   ./data/parler_distil/validation

Optional flags::

    --state_size 64        # SSM state per channel before Hedgehog doubling
                           # 64 (fast) / 256 (quality) / omit for N=D=1024
    --epochs 3
    --batch_size 4
    --lr 3e-4
    --warmup_steps 200
    --checkpoint_dir ./checkpoints/parler_mamba
    --config configs/parler_tts_to_mamba.yaml  # overrides individual flags

Prerequisites:
    1.  Run run_parler_preprocess.py first to build the .npz cache.
    2.  mlx-audio-train must be on the path (--mlx_audio_train).
"""
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx_audio_train", required=True)
    parser.add_argument("--train_cache", required=True)
    parser.add_argument("--val_cache",   required=True)
    parser.add_argument("--config", default=None,
                        help="Optional YAML config; CLI flags override it.")
    parser.add_argument("--state_size", type=int, default=None,
                        help="SSM state size. None=D (1024). Try 64 or 256.")
    parser.add_argument("--epochs",       type=int,   default=3)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int,   default=200)
    parser.add_argument("--max_audio_len",type=int,   default=512)
    parser.add_argument("--log_every",    type=int,   default=50)
    parser.add_argument("--eval_every",   type=int,   default=500)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/parler_mamba")
    parser.add_argument("--tb_log_dir",     default="runs/")
    args = parser.parse_args()

    # Optional YAML overrides (CLI flags win if both specified)
    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        s1 = cfg.get("stage1", {})
        st = cfg.get("student", {})
        args.epochs        = args.epochs        or s1.get("epochs", 3)
        args.batch_size    = args.batch_size    or s1.get("batch_size", 4)
        args.lr            = args.lr            or s1.get("learning_rate", 3e-4)
        args.warmup_steps  = args.warmup_steps  or s1.get("warmup_steps", 200)
        args.checkpoint_dir = args.checkpoint_dir or s1.get("checkpoint_dir", "./checkpoints/parler_mamba")
        args.state_size    = args.state_size    or st.get("state_size", None)

    sys.path.insert(0, args.mlx_audio_train)

    # ── Load teacher (IndicParlerTTS, original attention) ────────────────────
    import copy
    from models.indic_parler_tts.model import IndicParlerTTS
    print("[run_parler_stage1] Loading teacher from HuggingFace ...", flush=True)
    teacher = IndicParlerTTS.from_pretrained()
    teacher.freeze()

    # ── Build student (fresh backbone copy + SSM replacement) ───────────────
    from src.mlx.parler_model import ParlerMambaMLX
    print("[run_parler_stage1] Building ParlerMambaMLX student ...", flush=True)
    # Load a second instance for the student backbone
    student_base = IndicParlerTTS.from_pretrained()
    student = ParlerMambaMLX(student_base, state_size=args.state_size)
    print(student.param_summary(), flush=True)

    # ── Data loaders ────────────────────────────────────────────────────────
    from src.mlx.parler_data import make_parler_loaders
    train_loader, val_loader = make_parler_loaders(
        train_cache_dir=args.train_cache,
        val_cache_dir=args.val_cache,
        batch_size=args.batch_size,
        max_audio_len=args.max_audio_len,
    )
    print(
        f"[run_parler_stage1] "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} samples",
        flush=True,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    from src.mlx.parler_trainer import train_parler_mlx_stage1
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    losses = train_parler_mlx_stage1(
        student=student,
        teacher=teacher,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        tb_log_dir=args.tb_log_dir,
        log_every=args.log_every,
        eval_every=args.eval_every,
    )

    final = losses[-1] if losses else float("nan")
    print(f"[run_parler_stage1] Done. Final cosine loss: {final:.4f}", flush=True)
    print(
        f"[run_parler_stage1] Checkpoints saved to: {args.checkpoint_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
