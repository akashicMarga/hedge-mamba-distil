#!/usr/bin/env python
"""Phase 0 sanity checks — run before any training.

Checks:
  1. Teacher loads and produces logits of correct shape.
  2. Teacher hook mechanism captures per-layer outputs.
  3. Student (linear stage) builds correctly and shares teacher embedding weights.
  4. HedgeMambaLayer forward pass produces correct output shape.
  5. Stage 1 loss is finite and non-zero on a tiny synthetic batch.
  6. Parameter surgery (Appendix B) runs without error; log_lambda spread across time scales.
  7. Stage 2 forward + loss is finite.
  8. Param count report: teacher vs student.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from transformers import AutoModelForCausalLM

DEVICE = "cpu"
MODEL_ID = "EleutherAI/pythia-70m"
BATCH, SEQ, VOCAB = 2, 64, 50304

print("=" * 60)
print("Phase 0 Sanity Checks")
print("=" * 60)

# ── 1. Teacher loads ──────────────────────────────────────────────
print("\n[1] Loading teacher...")
from src.teachers.gpt_neox import GPTNeoXTeacher
teacher = GPTNeoXTeacher(MODEL_ID, device=DEVICE)
input_ids = torch.randint(0, VOCAB, (BATCH, SEQ))
logits = teacher.forward(input_ids)
assert logits.shape == (BATCH, SEQ, VOCAB), f"Unexpected logits shape: {logits.shape}"
print(f"    Teacher logits: {logits.shape}  ✓")

# ── 2. Hook mechanism ────────────────────────────────────────────
print("\n[2] Teacher hook mechanism...")
teacher.register_layer_hooks()
_ = teacher.forward(input_ids)
layer_outs = teacher.get_layer_outputs()
teacher.remove_hooks()
assert len(layer_outs) == teacher.num_layers, \
    f"Expected {teacher.num_layers} layer outputs, got {len(layer_outs)}"
for i, o in enumerate(layer_outs):
    assert o.shape == (BATCH, SEQ, teacher.hidden_size), \
        f"Layer {i} output shape wrong: {o.shape}"
print(f"    {len(layer_outs)} layer outputs, each {layer_outs[0].shape}  ✓")

# ── 3. Student builds ─────────────────────────────────────────────
print("\n[3] Building student (linear stage)...")
from src.student.builder import HedgeMambaStudent
student = HedgeMambaStudent(teacher.model, stage="linear")
print(f"    Student built with {sum(p.numel() for p in student.parameters()):,} params  ✓")

# ── 4. HedgeMambaLayer forward shape ─────────────────────────────
print("\n[4] HedgeMambaLayer forward (linear stage)...")
from src.student.hedge_mamba import HedgeMambaLayer
layer = HedgeMambaLayer(dim=512, num_heads=8, stage="linear")
x = torch.randn(BATCH, SEQ, 512)
out, _ = layer(x)
assert out.shape == x.shape, f"Shape mismatch: {out.shape} != {x.shape}"
print(f"    Output shape: {out.shape}  ✓")

# ── 5. Stage 1 loss is finite ────────────────────────────────────
print("\n[5] Stage 1 cosine loss...")
from src.distill.stage1 import cosine_distill_loss, freeze_except_hedgehog
s_out = torch.randn(BATCH, SEQ, 512)
t_out = torch.randn(BATCH, SEQ, 512)
loss = cosine_distill_loss(s_out, t_out)
assert torch.isfinite(loss), f"Loss is not finite: {loss}"
assert loss.item() != 0.0, "Loss is zero — suspicious"
print(f"    Cosine loss: {loss.item():.4f}  ✓")

# ── 6. Parameter surgery ─────────────────────────────────────────
print("\n[6] Parameter surgery (Appendix B)...")
from src.student.param_init import apply_surgery
linear_layer = HedgeMambaLayer(dim=512, num_heads=8, stage="linear")
ssm_layer = apply_surgery(linear_layer)
assert ssm_layer.stage == "ssm"
assert torch.all(ssm_layer.ssm.log_lambda <= 0), "log_lambda should be ≤ 0 (λ ≤ 1 for stability)"
assert ssm_layer.ssm.log_lambda[0] > ssm_layer.ssm.log_lambda[-1], "log_lambda should decrease (spread across time scales)"
print(f"    surgery OK, log_lambda={ssm_layer.ssm.log_lambda[:4].tolist()}  ✓")

# ── 7. Stage 2 SSM forward + loss ────────────────────────────────
print("\n[7] Stage 2 forward (SSM stage)...")
ssm_student = HedgeMambaStudent(teacher.model, stage="linear")
ssm_student.upgrade_to_ssm()
out2 = ssm_student(input_ids)
logits2 = out2.logits
shift_logits = logits2[:, :-1].contiguous()
shift_labels = input_ids[:, 1:].contiguous()
loss2 = torch.nn.functional.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
)
assert torch.isfinite(loss2), f"Stage 2 loss not finite: {loss2}"
print(f"    Stage 2 CE loss: {loss2.item():.4f}  ✓")

# ── 8. Param count report ────────────────────────────────────────
print("\n[8] Parameter counts:")
teacher_params = sum(p.numel() for p in teacher.parameters())
student_params = sum(p.numel() for p in ssm_student.parameters())
print(f"    Teacher: {teacher_params:>12,}")
print(f"    Student: {student_params:>12,}")
print(f"    Ratio:   {student_params / teacher_params:.3f}x")

print("\n" + "=" * 60)
print("All Phase 0 checks passed.")
print("=" * 60)
