#!/usr/bin/env python -u
"""Run Stage 1: Hedgehog feature map training."""
import argparse
import yaml
import torch
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.teachers.gpt_neox import GPTNeoXTeacher
from src.student.builder import HedgeMambaStudent
from src.distill.cache import cache_teacher_outputs, CachedTeacherDataset
from src.distill.stage1 import train_stage1
from src.data.owt import make_dataloader, make_fast_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia_70m_to_mamba.yaml")
    parser.add_argument("--debug", action="store_true", help="Use debug token budget")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-cache", action="store_true", help="Skip teacher caching (cache already exists)")
    parser.add_argument("--bin", default=None,
                        help="Path to pre-tokenised .bin file (fast loader). Falls back to streaming if not set.")
    parser.add_argument("--cache-batch-size", type=int, default=8)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    total_tokens = cfg["stage1"]["debug_tokens"] if args.debug else cfg["stage1"]["tokens"]

    print(f"Stage 1 | model={cfg['teacher']['model_id']} | device={device} | tokens={total_tokens:,}", flush=True)

    teacher = GPTNeoXTeacher(cfg["teacher"]["model_id"], device=device)

    cache_dir = cfg["stage1"]["teacher_cache_dir"]
    if not args.skip_cache:
        if args.bin:
            print(f"Caching teacher outputs from bin file (batch={args.cache_batch_size})...")
            cache_loader = make_fast_dataloader(
                args.bin,
                seq_len=cfg["data"]["seq_len"],
                batch_size=args.cache_batch_size,
                max_tokens=total_tokens,
                num_workers=4,
                pin_memory=False,
            )
        else:
            print(f"Caching teacher outputs (streaming, batch={args.cache_batch_size})...")
            cache_loader = make_dataloader(
                tokenizer_id=cfg["data"]["tokenizer"],
                seq_len=cfg["data"]["seq_len"],
                batch_size=args.cache_batch_size,
                max_tokens=total_tokens,
                num_workers=0,
            )
        cache_teacher_outputs(
            teacher, cache_loader, cache_dir,
            compile_teacher=not args.no_compile,
        )

    cached_ds = CachedTeacherDataset(cache_dir)
    cached_loader = torch.utils.data.DataLoader(
        cached_ds, batch_size=None, num_workers=2, pin_memory=(device != "cpu")
    )

    student = HedgeMambaStudent(teacher.model, stage="linear").to(device)
    optimizer = torch.optim.AdamW(
        [p for p in student.parameters() if p.requires_grad],
        lr=cfg["stage1"]["lr"],
    )

    losses = train_stage1(student, cached_loader, optimizer, device=device, total_tokens=total_tokens)

    ckpt_dir = Path(cfg["stage1"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(student.state_dict(), ckpt_dir / "stage1_final.pt")
    print(f"Stage 1 done. Final loss: {losses[-1]:.4f}", flush=True)


if __name__ == "__main__":
    main()
