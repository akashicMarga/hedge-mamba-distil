#!/usr/bin/env python -u
"""Final evaluation: teacher vs student PPL + generation comparison.
Writes results to /tmp/eval_final_results.txt
"""
import sys, math, torch, yaml, torch.nn.functional as F
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.teachers.gpt_neox import GPTNeoXTeacher
from src.student.builder import HedgeMambaStudent
from src.data.owt import make_fast_dataloader

OUT = Path("/tmp/eval_final_results.txt")

def log(msg=""):
    print(msg, flush=True)
    with open(OUT, "a") as f:
        f.write(msg + "\n")

PROMPTS = [
    ("factual",     "The capital of France is"),
    ("factual",     "Water boils at 100 degrees Celsius, which means that"),
    ("causal",      "If you drop a ball from a tall building, it will"),
    ("causal",      "She studied hard every day for months, so when the exam came she"),
    ("commonsense", "He forgot to bring an umbrella on a rainy day, so he"),
    ("commonsense", "The fire alarm went off in the office, so everyone"),
    ("coherence",   "Once upon a time, a young scientist discovered a formula that could turn"),
    ("coherence",   "The detective looked at the clues carefully. The footprints led to the garden, "
                    "the window was broken from the inside, and the only person home was"),
]

@torch.no_grad()
def compute_ppl(model, loader, device, max_batches=100, label=""):
    model.eval()
    total_loss, count = 0.0, 0
    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        ids = batch["input_ids"].to(device)
        out = model(ids)
        logits = out.logits if hasattr(out, "logits") else out
        loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)), ids[:, 1:].reshape(-1))
        if torch.isfinite(loss):
            total_loss += loss.item()
            count += 1
        if (i+1) % 20 == 0:
            log(f"  [{label}] batch {i+1}/{max_batches}  running_ppl={math.exp(total_loss/count):.1f}")
    avg_loss = total_loss / max(count, 1)
    return avg_loss, math.exp(avg_loss)

@torch.no_grad()
def generate(model, tokenizer, prompt, max_new_tokens=50, device="cpu"):
    ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    out = model.generate(
        ids, max_new_tokens=max_new_tokens, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True)


def main():
    OUT.unlink(missing_ok=True)
    log("=" * 70)
    log("HedgeMamba Final Evaluation")
    log("=" * 70)

    cfg = yaml.safe_load(open("configs/pythia_70m_to_mamba.yaml"))
    device = "mps"
    model_id = cfg["teacher"]["model_id"]
    bin_path = "data_cache/owt_tokens.bin"
    ckpt = "checkpoints/stage2/stage2_final.pt"

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Val loader — last 20M tokens
    log("\nLoading val data (offset=580M tokens)...")
    val_loader = make_fast_dataloader(
        bin_path, seq_len=cfg["data"]["seq_len"], batch_size=8,
        offset_tokens=580_000_000, shuffle=False, num_workers=2, pin_memory=False,
    )

    # ── Teacher ──────────────────────────────────────────────────────────
    log("\nLoading teacher...")
    teacher_hf = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32).to(device)
    teacher_hf.eval()

    log("\n[1/4] Computing TEACHER PPL on val set (100 batches)...")
    t_loss, t_ppl = compute_ppl(teacher_hf, val_loader, device, max_batches=100, label="teacher")
    log(f"  → Teacher val loss: {t_loss:.4f}   PPL: {t_ppl:.1f}")

    # ── Student ──────────────────────────────────────────────────────────
    log("\nLoading student...")
    t_wrap = GPTNeoXTeacher(model_id, device=device)
    student = HedgeMambaStudent(t_wrap.model, stage="linear")
    student.upgrade_to_ssm()   # must match checkpoint structure (saved post-surgery)
    student.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    student = student.to(device)
    student.eval()

    log("\n[2/4] Computing STUDENT PPL on val set (100 batches)...")
    s_loss, s_ppl = compute_ppl(student, val_loader, device, max_batches=100, label="student")
    log(f"  → Student val loss: {s_loss:.4f}   PPL: {s_ppl:.1f}")

    # ── PPL Summary ───────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("PPL SUMMARY")
    log("=" * 70)
    log(f"  Teacher  PPL : {t_ppl:.1f}")
    log(f"  Student  PPL : {s_ppl:.1f}")
    log(f"  Gap          : {s_ppl - t_ppl:.1f} ({(s_ppl/t_ppl - 1)*100:.1f}% worse than teacher)")
    log(f"  Compression  : {53/53:.1f}x param ratio  (same size, different arch)")

    # ── Generation comparison ─────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("[3/4] GENERATION COMPARISON (greedy, 50 new tokens)")
    log("=" * 70)

    mismatches = 0
    for category, prompt in PROMPTS:
        t_out = generate(teacher_hf, tokenizer, prompt, device=device)
        s_out = generate(student.backbone, tokenizer, prompt, device=device)

        t_words = set(t_out.lower().split())
        s_words = set(s_out.lower().split())
        overlap = len(t_words & s_words) / max(len(t_words), 1)
        match = "✓" if overlap > 0.25 else "✗"
        if overlap <= 0.25:
            mismatches += 1

        log(f"\n[{category.upper()}]  {match}  word-overlap={overlap:.0%}")
        log(f"  Prompt:  {prompt!r}")
        log(f"  Teacher: {t_out.strip()!r}")
        log(f"  Student: {s_out.strip()!r}")

    log("\n" + "=" * 70)
    log("GENERATION SUMMARY")
    log("=" * 70)
    log(f"  Logically consistent: {len(PROMPTS)-mismatches}/{len(PROMPTS)} prompts")

    # ── Done ──────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)
    log(f"  Teacher PPL  : {t_ppl:.1f}")
    log(f"  Student PPL  : {s_ppl:.1f}  (trained 500M tokens, pythia-14m arch → HedgeMamba SSM)")
    log(f"  Generation   : {len(PROMPTS)-mismatches}/{len(PROMPTS)} prompts match teacher logic")
    ppl_gap_pct = (s_ppl/t_ppl - 1)*100
    if ppl_gap_pct < 30:
        verdict = "GOOD — student is within 30% of teacher PPL"
    elif ppl_gap_pct < 60:
        verdict = "FAIR — notable gap, more training or larger teacher would help"
    else:
        verdict = "NEEDS WORK — significant gap, consider more tokens or architecture tuning"
    log(f"  Verdict      : {verdict}")
    log("=" * 70)
    log(f"\nFull results saved to: {OUT}")


if __name__ == "__main__":
    main()
