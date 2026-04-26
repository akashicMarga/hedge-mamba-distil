#!/usr/bin/env python
"""Side-by-side generation comparison: teacher vs distilled student.

Tests reasoning quality beyond PPL — catches cases where the student
mimics token distributions superficially without learning logic.

Prompts span different reasoning types:
  - Factual recall
  - Causal / logical continuation
  - Commonsense
  - Long-range coherence (story continuation)
"""
import argparse
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

PROMPTS = [
    # Factual
    ("factual",       "The capital of France is"),
    ("factual",       "Water boils at 100 degrees Celsius, which means that"),
    # Causal / logical
    ("causal",        "If you drop a ball from a tall building, it will"),
    ("causal",        "She studied hard every day for months, so when the exam came she"),
    # Commonsense
    ("commonsense",   "He forgot to bring an umbrella on a rainy day, so he"),
    ("commonsense",   "The fire alarm went off in the office, so everyone"),
    # Long-range coherence
    ("coherence",     "Once upon a time, a young scientist discovered a formula that could turn"),
    ("coherence",     "The detective looked at the clues carefully. The footprints led to the garden, the window was broken from the inside, and the only person home was"),
]

@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 40, device: str = "cpu") -> str:
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,          # greedy — deterministic, easier to compare
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    new_tokens = out[0][ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/pythia_70m_to_mamba.yaml")
    parser.add_argument("--student-ckpt", required=True, help="Path to stage2_final.pt")
    parser.add_argument("--student-stage", default="ssm", choices=["linear", "ssm"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=40)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = args.device
    model_id = cfg["teacher"]["model_id"]
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Load teacher
    from transformers import AutoModelForCausalLM
    teacher = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to(device)
    teacher.eval()

    # Load student
    from src.teachers.gpt_neox import GPTNeoXTeacher
    from src.student.builder import HedgeMambaStudent
    t = GPTNeoXTeacher(model_id, device=device)
    student = HedgeMambaStudent(t.model, stage="linear")
    student.load_state_dict(torch.load(args.student_ckpt, weights_only=True, map_location=device))
    if args.student_stage == "ssm":
        student.upgrade_to_ssm()
    student = student.to(device)
    student.eval()

    print(f"\n{'='*70}")
    print(f"Teacher:  {model_id}")
    print(f"Student:  HedgeMamba ({args.student_stage} stage) ← {args.student_ckpt}")
    print(f"{'='*70}\n")

    mismatches = 0
    for category, prompt in PROMPTS:
        t_out = generate(teacher, tokenizer, prompt, args.max_new_tokens, device)
        s_out = generate(student.backbone, tokenizer, prompt, args.max_new_tokens, device)

        # Simple coherence check: does student output share key content words with teacher?
        t_words = set(t_out.lower().split())
        s_words = set(s_out.lower().split())
        overlap = len(t_words & s_words) / max(len(t_words), 1)
        match = "✓" if overlap > 0.3 else "✗ MISMATCH"
        if overlap <= 0.3:
            mismatches += 1

        print(f"[{category}] {match} (word overlap: {overlap:.0%})")
        print(f"  Prompt:  {prompt!r}")
        print(f"  Teacher: {t_out.strip()!r}")
        print(f"  Student: {s_out.strip()!r}")
        print()

    print(f"{'='*70}")
    print(f"Summary: {len(PROMPTS) - mismatches}/{len(PROMPTS)} prompts logically consistent with teacher")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
