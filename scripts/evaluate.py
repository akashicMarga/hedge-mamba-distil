#!/usr/bin/env python
"""Evaluate a checkpoint: PPL + lm-eval tasks + optional edge bench."""
import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.teachers.gpt_neox import GPTNeoXTeacher
from src.student.builder import HedgeMambaStudent
from src.eval.perplexity import compute_perplexity
from src.data.owt import make_dataloader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia_70m_to_mamba.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--stage", choices=["linear", "ssm"], default="ssm")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--ppl-only", action="store_true")
    parser.add_argument("--edge-bench", action="store_true")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    teacher = GPTNeoXTeacher(cfg["teacher"]["model_id"], device=device)
    student = HedgeMambaStudent(teacher.model, stage=args.stage)
    student.load_state_dict(torch.load(args.ckpt, weights_only=True, map_location=device))
    student = student.to(device)

    val_loader = make_dataloader(
        tokenizer_id=cfg["data"]["tokenizer"],
        seq_len=cfg["data"]["seq_len"],
        batch_size=8,
        split="train",  # OWT has no official val split; use first N batches
        max_tokens=5_000_000,
    )

    ppl = compute_perplexity(student, val_loader, device=device, max_batches=200)
    print(f"Val PPL: {ppl:.3f}")

    if not args.ppl_only:
        from src.eval.lm_eval_wrapper import evaluate_tasks
        tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"])
        results = evaluate_tasks(student, tokenizer, cfg["eval"]["tasks"], device=device)
        for task, metrics in results.items():
            print(f"{task}: {metrics}")

    if args.edge_bench:
        from src.eval.edge_bench import bench_model
        tokenizer = AutoTokenizer.from_pretrained(cfg["data"]["tokenizer"])
        bench_model(student, tokenizer, device=device)


if __name__ == "__main__":
    main()
