#!/usr/bin/env python -u
"""WhisperMamba: full two-stage distillation pipeline.

Stage 1 — Cosine distillation
  Each Mamba SSM layer is trained to mimic the corresponding Whisper decoder
  self-attention output (cosine loss).  No transcript supervision yet.
  Result: SSM layers are warm-initialized to approximate Whisper's attention.

Stage 2 — ASR fine-tuning
  End-to-end cross-entropy training on transcripts (teacher-forced).
  Frozen: encoder, embeddings (same as before).
  Logs: train/loss, val/loss, val/wer (no PPL — meaningless for ASR).

Architecture:
  Encoder : frozen Whisper-tiny encoder (Conv + 4 transformer layers)
  Decoder : 4 HedgeMamba SSM layers + cross-attention + FFN (from Whisper)

Quick debug run (~5 min, skips Stage 1):
  python scripts/run_whisper.py --device mps --debug --skip_stage1

Full run (LibriSpeech train.100, Stage 1 × 2 epochs + Stage 2 × 5 epochs):
  python scripts/run_whisper.py --device mps
  python scripts/run_whisper.py --device cuda   # NVIDIA GPU (~10× faster)

Resume from Stage 1 checkpoint (skip straight to Stage 2):
  python scripts/run_whisper.py --skip_stage1 --resume_stage2 checkpoints/whisper_mamba/stage1_epoch_2.pt
"""
import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from transformers import WhisperForConditionalGeneration
from src.student.whisper_mamba import WhisperMambaStudent
from src.data.librispeech import make_librispeech_loaders
from src.distill.whisper_stage1 import train_whisper_stage1, _ssm_params
from src.distill.whisper_train import train_whisper_mamba, compute_wer


def print_sample_transcriptions(model, val_loader, processor, device, n=3):
    model.eval()
    sample = next(iter(val_loader))
    input_features = sample["input_features"][:n].to(device)
    labels = sample["labels"][:n]

    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="en",
            task="transcribe",
            max_new_tokens=128,
            repetition_penalty=1.1,
        )

    print("\n=== Sample Transcriptions ===", flush=True)
    for i in range(min(n, len(predicted_ids))):
        pred = processor.decode(predicted_ids[i], skip_special_tokens=True)
        ref = processor.decode(labels[i].clamp(min=0), skip_special_tokens=True)
        print(f"  [{i+1}] Ref:  {ref!r}", flush=True)
        print(f"       Pred: {pred!r}", flush=True)
    model.train()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/whisper_tiny_to_mamba.yaml")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tiny subset (100 train, 20 val) for quick sanity check",
    )
    parser.add_argument(
        "--skip_stage1",
        action="store_true",
        help="Skip Stage 1 cosine distillation (use if resuming from a Stage 1 checkpoint)",
    )
    parser.add_argument(
        "--resume_stage1",
        default=None,
        help="Path to Stage 1 checkpoint to resume Stage 1 from",
    )
    parser.add_argument(
        "--resume_stage2",
        default=None,
        help="Path to checkpoint to start Stage 2 from (implies --skip_stage1)",
    )
    args = parser.parse_args()

    if args.resume_stage2:
        args.skip_stage1 = True

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    model_id = cfg["teacher"]["model_id"]

    print(f"WhisperMamba | teacher={model_id} | device={device}", flush=True)
    print(f"Architecture : frozen Whisper encoder + Mamba SSM decoder", flush=True)

    # ── Load Whisper teacher ──────────────────────────────────────────────────
    print("\nLoading Whisper teacher...", flush=True)
    teacher = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float32
    )

    # ── Build student ─────────────────────────────────────────────────────────
    print("Building WhisperMamba student...", flush=True)
    student = WhisperMambaStudent(teacher, state_size=cfg["student"]["state_size"])
    print(student.param_summary(), flush=True)

    # ── Data ──────────────────────────────────────────────────────────────────
    train_loader, val_loader, processor = make_librispeech_loaders(
        model_id=model_id,
        language=cfg["teacher"]["language"],
        task=cfg["teacher"]["task"],
        train_split=cfg["data"]["train_split"],
        val_split=cfg["data"]["val_split"],
        batch_size=cfg["training"]["batch_size"],
        max_label_length=cfg["data"]["max_label_length"],
    )

    if args.debug:
        from torch.utils.data import Subset, DataLoader as DL
        from src.data.librispeech import LibriSpeechCollator
        collate_fn = LibriSpeechCollator()
        train_loader = DL(
            Subset(train_loader.dataset, range(100)),
            batch_size=cfg["training"]["batch_size"],
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=0,
        )
        val_loader = DL(
            Subset(val_loader.dataset, range(20)),
            batch_size=cfg["training"]["batch_size"],
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
        )
        print("DEBUG mode: 100 train / 20 val samples", flush=True)

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])

    # ── Baseline WER ─────────────────────────────────────────────────────────
    student = student.to(device)
    print("\n[Baseline] Computing WER before any training...", flush=True)
    baseline_wer = compute_wer(student, val_loader, processor, device, max_batches=5)
    if baseline_wer is not None:
        print(f"  Baseline WER: {baseline_wer*100:.1f}%", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 1 — cosine distillation
    # ══════════════════════════════════════════════════════════════════════════
    if not args.skip_stage1:
        print("\n" + "█"*55, flush=True)
        print("  STAGE 1 — Layer-wise cosine distillation", flush=True)
        print("█"*55, flush=True)

        if args.resume_stage1:
            student.load_state_dict(
                torch.load(args.resume_stage1, weights_only=True, map_location="cpu")
            )
            print(f"Stage 1 resumed from: {args.resume_stage1}", flush=True)

        # Stage 1 optimizer: only SSM params, higher lr (cosine loss landscape is smooth)
        s1_optimizer = torch.optim.AdamW(
            _ssm_params(student),
            lr=cfg["training"].get("stage1_lr", 5e-4),
            weight_decay=0.01,
        )
        s1_epochs = cfg["training"].get("stage1_epochs", 2) if not args.debug else 1

        s1_warmup = 200
        s1_scheduler = torch.optim.lr_scheduler.LambdaLR(
            s1_optimizer, lambda step: min(1.0, step / s1_warmup)
        )

        s1_losses = train_whisper_stage1(
            teacher_model=teacher,
            student_model=student,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=s1_optimizer,
            device=device,
            epochs=s1_epochs,
            checkpoint_dir=str(ckpt_dir),
            tb_log_dir=cfg["training"]["tb_log_dir"],
            log_every=cfg["training"]["log_every"],
            eval_every=cfg["training"]["eval_every_steps"],
            scheduler=s1_scheduler,
        )

        # Save final Stage 1 checkpoint
        s1_final = ckpt_dir / "stage1_final.pt"
        torch.save(student.state_dict(), s1_final)
        print(f"\nStage 1 done. Final cos_loss: {s1_losses[-1]:.4f}", flush=True)
        print(f"Stage 1 checkpoint → {s1_final}", flush=True)

        # WER after Stage 1 (sanity check — should still be bad but not "a a a")
        print("\n[After Stage 1] WER check...", flush=True)
        s1_wer = compute_wer(student, val_loader, processor, device, max_batches=5)
        if s1_wer is not None:
            print(f"  WER after Stage 1: {s1_wer*100:.1f}%", flush=True)

    else:
        print("\n[Skipping Stage 1]", flush=True)
        if args.resume_stage2:
            student.load_state_dict(
                torch.load(args.resume_stage2, weights_only=True, map_location="cpu")
            )
            print(f"Stage 2 starting from checkpoint: {args.resume_stage2}", flush=True)

    # ══════════════════════════════════════════════════════════════════════════
    #  STAGE 2 — ASR fine-tuning with cross-entropy
    # ══════════════════════════════════════════════════════════════════════════
    print("\n" + "█"*55, flush=True)
    print("  STAGE 2 — ASR fine-tuning (cross-entropy + WER)", flush=True)
    print("█"*55, flush=True)

    s2_optimizer = torch.optim.AdamW(
        student.trainable_params(),
        lr=cfg["training"]["lr"],
        weight_decay=0.01,
    )
    s2_warmup = 500
    s2_scheduler = torch.optim.lr_scheduler.LambdaLR(
        s2_optimizer, lambda step: min(1.0, step / s2_warmup)
    )

    s2_losses = train_whisper_mamba(
        student_model=student,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=s2_optimizer,
        processor=processor,
        device=device,
        epochs=cfg["training"]["epochs"] if not args.debug else 1,
        checkpoint_dir=str(ckpt_dir),
        tb_log_dir=cfg["training"]["tb_log_dir"],
        log_every=cfg["training"]["log_every"],
        eval_every=cfg["training"]["eval_every_steps"],
        wer_every=cfg["training"].get("wer_every_steps", 1000),
        ss_max_p=cfg["training"].get("ss_max_p", 0.5),
        scheduler=s2_scheduler,
    )

    # ── Save final ────────────────────────────────────────────────────────────
    final_ckpt = ckpt_dir / "whisper_mamba_final.pt"
    torch.save(student.state_dict(), final_ckpt)
    print(f"\nFinal checkpoint → {final_ckpt}", flush=True)

    # ── Final WER ─────────────────────────────────────────────────────────────
    print("\n[Final] Computing WER after full training...", flush=True)
    final_wer = compute_wer(student, val_loader, processor, device, max_batches=20)
    if final_wer is not None:
        print(f"  Final WER: {final_wer*100:.1f}%", flush=True)

    print_sample_transcriptions(student, val_loader, processor, device)

    print(f"\nDone. Final Stage 2 loss: {s2_losses[-1]:.4f}", flush=True)
    if final_wer is not None:
        print(f"Final WER: {final_wer*100:.1f}%", flush=True)


if __name__ == "__main__":
    main()
