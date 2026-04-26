#!/usr/bin/env python -u
"""MLX two-stage WhisperMamba distillation pipeline.

Equivalent to scripts/run_whisper.py but runs entirely in MLX (no PyTorch
device usage at training time).  Estimated ~2.3× faster than PyTorch MPS.

Stage 1 — Cosine distillation:
  SSM layers warm-initialized to mimic Whisper decoder block outputs.
  Only HedgeMambaMixerMLX weights are trained.

Stage 2 — ASR fine-tuning:
  Cross-entropy + scheduled sampling (0 → 50% self-generated tokens).
  Trained: SSM, cross-attention, FFN, layer norms.

Quick debug run (~5 min, skips Stage 1):
  python scripts/run_mlx.py --debug --skip_stage1

Full run (~3.5 h on M-series):
  python scripts/run_mlx.py

Resume from Stage 1 checkpoint:
  python scripts/run_mlx.py --skip_stage1 --resume checkpoints/mlx/stage1_epoch_2.npz

Bridge from existing PyTorch checkpoint:
  python scripts/run_mlx.py --skip_stage1 --pt_ckpt checkpoints/whisper_mamba/stage1_final.pt
"""
import argparse
import sys
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import mlx.core as mx
import mlx_whisper.load_models as lm

from src.mlx.model import WhisperMambaMLX
from src.mlx.data import make_mlx_loaders
from src.mlx.trainer import train_mlx_stage1, train_mlx_stage2
from src.mlx.checkpoint import save as ckpt_save, load as ckpt_load, load_from_pt_checkpoint
from src.mlx.utils import compute_wer_mlx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      default="configs/whisper_tiny_to_mamba.yaml")
    parser.add_argument("--mlx_repo",   default="mlx-community/whisper-tiny-mlx")
    parser.add_argument("--debug",      action="store_true",
                        help="100 train / 20 val samples (~5 min sanity check)")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--resume",      default=None,
                        help="Path to .npz checkpoint (native MLX) to resume from")
    parser.add_argument("--pt_ckpt",     default=None,
                        help="Path to PyTorch .pt checkpoint to bridge into MLX student")
    args = parser.parse_args()

    if args.resume or args.pt_ckpt:
        args.skip_stage1 = True

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    tr = cfg["training"]
    state_size = cfg["student"]["state_size"]
    model_id   = cfg["teacher"]["model_id"]
    ckpt_dir   = Path(tr["checkpoint_dir"].replace("whisper_mamba", "mlx"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"WhisperMamba MLX | teacher={model_id} | state_size={state_size}", flush=True)

    # ── Build student ─────────────────────────────────────────────────────────
    print("\nBuilding WhisperMambaMLX student...", flush=True)
    student = WhisperMambaMLX(mlx_repo=args.mlx_repo, state_size=state_size)
    print(student.param_summary(), flush=True)

    # ── Teacher (unpatched mlx-whisper, frozen entirely) ──────────────────────
    print("Loading MLX teacher...", flush=True)
    teacher = lm.load_model(args.mlx_repo)
    teacher.freeze()

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, processor = make_mlx_loaders(
        model_id=model_id,
        language=cfg["teacher"]["language"],
        task=cfg["teacher"]["task"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        batch_size=tr["batch_size"],
        max_label_length=cfg["data"]["max_label_length"],
        debug=args.debug,
    )

    # ── Load checkpoint if requested ──────────────────────────────────────────
    if args.pt_ckpt:
        load_from_pt_checkpoint(student, args.pt_ckpt)
    elif args.resume:
        ckpt_load(student, args.resume)

    # ── Baseline WER ─────────────────────────────────────────────────────────
    print("\n[Baseline] WER before training...", flush=True)
    baseline_wer = compute_wer_mlx(student, val_loader, processor, max_batches=5)
    if baseline_wer is not None:
        print(f"  Baseline WER: {baseline_wer * 100:.1f}%", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 1
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_stage1:
        print("\n" + "█" * 55, flush=True)
        print("  STAGE 1 — Cosine distillation (MLX)", flush=True)
        print("█" * 55, flush=True)

        s1_epochs = tr.get("stage1_epochs", 2) if not args.debug else 1
        s1_losses = train_mlx_stage1(
            student=student,
            teacher=teacher,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=s1_epochs,
            lr=tr.get("stage1_lr", 5e-4),
            warmup_steps=200,
            checkpoint_dir=str(ckpt_dir),
            tb_log_dir=tr["tb_log_dir"],
            log_every=tr["log_every"],
            eval_every=tr["eval_every_steps"],
        )

        ckpt_save(
            student, str(ckpt_dir / "stage1_final"),
            meta={"stage": 1, "final_cos_loss": s1_losses[-1] if s1_losses else None},
        )
        print(f"\nStage 1 done.  Final cos_loss: {s1_losses[-1]:.4f}", flush=True)

        print("\n[After Stage 1] WER check...", flush=True)
        s1_wer = compute_wer_mlx(student, val_loader, processor, max_batches=5)
        if s1_wer is not None:
            print(f"  WER after Stage 1: {s1_wer * 100:.1f}%", flush=True)
    else:
        print("\n[Skipping Stage 1]", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 2
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "█" * 55, flush=True)
    print("  STAGE 2 — ASR fine-tuning (cross-entropy + SS)", flush=True)
    print("█" * 55, flush=True)

    s2_losses = train_mlx_stage2(
        student=student,
        train_loader=train_loader,
        val_loader=val_loader,
        processor=processor,
        epochs=tr["epochs"] if not args.debug else 1,
        lr=tr["lr"],
        warmup_steps=500,
        checkpoint_dir=str(ckpt_dir),
        tb_log_dir=tr["tb_log_dir"],
        log_every=tr["log_every"],
        eval_every=tr["eval_every_steps"],
        wer_every=tr.get("wer_every_steps", 1000),
        ss_max_p=tr.get("ss_max_p", 0.5),
    )

    # ── Save final ────────────────────────────────────────────────────────────
    ckpt_save(
        student, str(ckpt_dir / "whisper_mamba_mlx_final"),
        meta={"stage": 2, "final_loss": s2_losses[-1] if s2_losses else None},
    )

    # ── Final WER ─────────────────────────────────────────────────────────────
    print("\n[Final] WER after full training...", flush=True)
    final_wer = compute_wer_mlx(student, val_loader, processor, max_batches=20)
    if final_wer is not None:
        print(f"  Final WER: {final_wer * 100:.1f}%", flush=True)

    print(f"\nDone.  Final Stage 2 loss: {s2_losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
