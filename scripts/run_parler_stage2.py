#!/usr/bin/env python -u
"""Stage 2: 9-codebook CE fine-tuning with scheduled sampling.

Usage::

    python scripts/run_parler_stage2.py \
        --mlx_audio_train /path/to/mlx-audio-train \
        --train_cache ./data/parler_distil/train \
        --val_cache   ./data/parler_distil/validation \
        --stage1_ckpt ./checkpoints/parler_mamba/stage1_epoch_3

Optional flags::

    --state_size 64
    --epochs 5
    --batch_size 4
    --lr 1e-4
    --warmup_steps 500
    --ss_max_p 0.5      # max fraction of tokens replaced by student preds
    --checkpoint_dir ./checkpoints/parler_mamba
    --config configs/parler_tts_to_mamba.yaml

Loads stage1 checkpoint when --stage1_ckpt is given; otherwise starts from
the teacher-initialized weights (useful for ablations).
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
    parser.add_argument("--stage1_ckpt", default=None,
                        help="Path prefix to stage1 checkpoint (without .npz/.json)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--state_size",   type=int,   default=None)
    parser.add_argument("--epochs",       type=int,   default=5)
    parser.add_argument("--batch_size",   type=int,   default=4)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int,   default=500)
    parser.add_argument("--ss_max_p",     type=float, default=0.5)
    parser.add_argument("--max_audio_len",type=int,   default=512)
    parser.add_argument("--log_every",    type=int,   default=50)
    parser.add_argument("--eval_every",   type=int,   default=500)
    parser.add_argument("--checkpoint_dir", default="./checkpoints/parler_mamba")
    parser.add_argument("--tb_log_dir",     default="runs/")
    args = parser.parse_args()

    if args.config:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        s2 = cfg.get("stage2", {})
        st = cfg.get("student", {})
        args.epochs        = args.epochs        or s2.get("epochs", 5)
        args.batch_size    = args.batch_size    or s2.get("batch_size", 4)
        args.lr            = args.lr            or s2.get("learning_rate", 1e-4)
        args.warmup_steps  = args.warmup_steps  or s2.get("warmup_steps", 500)
        args.ss_max_p      = args.ss_max_p      or s2.get("ss_max_p", 0.5)
        args.checkpoint_dir = args.checkpoint_dir or s2.get("checkpoint_dir", "./checkpoints/parler_mamba")
        args.state_size    = args.state_size    or st.get("state_size", None)

    sys.path.insert(0, args.mlx_audio_train)

    # ── Build student backbone ───────────────────────────────────────────────
    from models.indic_parler_tts.model import IndicParlerTTS
    from src.mlx.parler_model import ParlerMambaMLX
    from src.mlx import parler_checkpoint as ckpt_io

    print("[run_parler_stage2] Loading student backbone ...", flush=True)
    student_base = IndicParlerTTS.from_pretrained()
    student = ParlerMambaMLX(student_base, state_size=args.state_size)

    if args.stage1_ckpt:
        print(f"[run_parler_stage2] Loading stage1 checkpoint: {args.stage1_ckpt}",
              flush=True)
        ckpt_io.load(student, args.stage1_ckpt)
    else:
        print("[run_parler_stage2] No stage1 checkpoint — starting from scratch",
              flush=True)

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
        f"[run_parler_stage2] "
        f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} samples",
        flush=True,
    )

    # ── Train ────────────────────────────────────────────────────────────────
    from src.mlx.parler_trainer import train_parler_mlx_stage2
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    losses = train_parler_mlx_stage2(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        warmup_steps=args.warmup_steps,
        checkpoint_dir=args.checkpoint_dir,
        tb_log_dir=args.tb_log_dir,
        log_every=args.log_every,
        eval_every=args.eval_every,
        ss_max_p=args.ss_max_p,
    )

    final = losses[-1] if losses else float("nan")
    print(f"[run_parler_stage2] Done. Final CE loss: {final:.4f}", flush=True)
    print(
        f"[run_parler_stage2] Checkpoints saved to: {args.checkpoint_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
