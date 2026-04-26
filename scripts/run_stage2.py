#!/usr/bin/env python -u
"""Run Stage 2: HedgeMamba fine-tuning with cross-entropy."""
import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.teachers.gpt_neox import GPTNeoXTeacher
from src.student.builder import HedgeMambaStudent
from src.distill.stage2 import train_stage2
from src.data.owt import make_dataloader, make_fast_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia_70m_to_mamba.yaml")
    parser.add_argument("--stage1-ckpt", required=True, help="Path to stage1_final.pt")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--bin", default=None,
                        help="Path to pre-tokenised .bin file (fast loader). Falls back to streaming if not set.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    total_tokens = cfg["stage2"]["debug_tokens"] if args.debug else cfg["stage2"]["tokens"]

    print(f"Stage 2 | model={cfg['teacher']['model_id']} | device={device} | tokens={total_tokens:,}", flush=True)

    teacher = GPTNeoXTeacher(cfg["teacher"]["model_id"], device=device)
    student = HedgeMambaStudent(teacher.model, stage="linear")
    student.load_state_dict(torch.load(args.stage1_ckpt, weights_only=True))

    print("Applying parameter surgery (Appendix B)...", flush=True)
    student.upgrade_to_ssm()
    student = student.to(device)

    val_loader = None
    if args.bin:
        print(f"Using fast bin loader: {args.bin}", flush=True)
        # Train: first 580M tokens; Val: last ~20M tokens
        train_tokens = cfg["stage2"].get("train_split_tokens", 580_000_000)
        dataloader = make_fast_dataloader(
            args.bin,
            seq_len=cfg["data"]["seq_len"],
            batch_size=cfg["stage2"]["batch_size"],
            max_tokens=train_tokens,
            num_workers=4,
            pin_memory=False,
            shuffle=True,
        )
        val_loader = make_fast_dataloader(
            args.bin,
            seq_len=cfg["data"]["seq_len"],
            batch_size=cfg["stage2"]["batch_size"],
            offset_tokens=train_tokens,
            num_workers=2,
            pin_memory=False,
            shuffle=False,
        )
        print(f"Val loader: offset={train_tokens:,} tokens", flush=True)
    else:
        dataloader = make_dataloader(
            tokenizer_id=cfg["data"]["tokenizer"],
            seq_len=cfg["data"]["seq_len"],
            batch_size=cfg["stage2"]["batch_size"],
            max_tokens=total_tokens,
            num_workers=0,
        )

    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg["stage2"]["lr"],
    )

    losses = train_stage2(
        student,
        dataloader,
        optimizer,
        device=device,
        total_tokens=total_tokens,
        checkpoint_every_tokens=cfg["stage2"]["checkpoint_every_tokens"],
        checkpoint_dir=cfg["stage2"]["checkpoint_dir"],
        val_loader=val_loader,
        tb_log_dir="runs/",
        eval_every_tokens=cfg["stage2"].get("eval_every_tokens", 10_000_000),
    )

    ckpt_dir = Path(cfg["stage2"]["checkpoint_dir"])
    torch.save(student.state_dict(), ckpt_dir / "stage2_final.pt")
    print(f"Stage 2 done. Final loss: {losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
