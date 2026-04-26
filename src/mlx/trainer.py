"""MLX training loops for WhisperMamba distillation.

Stage 1 — cosine distillation (SSM warm-up):
  student.freeze_for_stage1()
  Each SSM block is trained to mimic the teacher decoder block output.
  Loss: mean cosine distance across blocks.

Stage 2 — ASR fine-tuning (cross-entropy + scheduled sampling):
  student.freeze_for_stage2()
  Loss: cross-entropy on next transcript token.
  Scheduled sampling: gradually replaces GT tokens with model predictions
  (0 → ss_max_p over first half of training) to close teacher-forcing gap.

MLX training pattern:
  loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
  loss, grads = loss_and_grad_fn(model, *args)
  optimizer.update(model, grads)
  mx.eval(model.parameters(), optimizer.state, loss)  # flush lazy graph
"""
import math
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.mlx.loss import cosine_distill_loss, ce_loss
from src.mlx.model import get_teacher_hiddens
from src.mlx.utils import (
    cosine_lr_with_warmup, clip_grad_norm,
    compute_val_loss_mlx, compute_wer_mlx,
)
from src.mlx.data import pt_batch_to_mlx
from src.mlx import checkpoint as ckpt_io

# Whisper-tiny decoder_start_token_id — prepended to every decoder input sequence
_WHISPER_BOS = 50258


# ── Stage 1 ───────────────────────────────────────────────────────────────────

def train_mlx_stage1(
    student,
    teacher,
    train_loader,
    val_loader,
    epochs: int = 2,
    lr: float = 5e-4,
    warmup_steps: int = 200,
    checkpoint_dir: str = "./checkpoints/mlx",
    tb_log_dir: str = "runs/",
    log_every: int = 50,
    eval_every: int = 500,
) -> list[float]:
    """Stage 1: train SSM layers to mimic Whisper decoder block outputs.

    Teacher is frozen entirely.  Only student.block.attn (HedgeMambaMixerMLX)
    receives gradients (enforced by freeze_for_stage1).
    """
    student.freeze_for_stage1()
    teacher.freeze()
    print(f"[Stage 1] {student.param_summary()}", flush=True)

    run_name = datetime.now().strftime("mlx_stage1_%Y%m%d_%H%M%S")
    log_dir = Path(tb_log_dir) / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[Stage 1] TensorBoard → {log_dir}", flush=True)

    total_steps = epochs * len(train_loader)
    optimizer = optim.Adam(learning_rate=lr)

    # Loss function — teacher_hiddens are pre-computed outside the grad fn
    def s1_loss_fn(model, mel, decoder_ids, teacher_hiddens):
        _, student_hiddens = model(mel, decoder_ids)
        return cosine_distill_loss(student_hiddens, teacher_hiddens)

    loss_and_grad_fn = nn.value_and_grad(student, s1_loss_fn)

    losses: list[float] = []
    global_step = 0
    last_val_loss = float("nan")

    for epoch in range(epochs):
        print(f"\n{'='*55}", flush=True)
        print(f"[Stage 1] Epoch {epoch + 1} / {epochs}", flush=True)
        print(f"{'='*55}", flush=True)

        for batch in tqdm(train_loader, desc=f"[S1] Epoch {epoch+1}"):
            # LR schedule
            optimizer.learning_rate = cosine_lr_with_warmup(
                global_step, lr_max=lr, total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

            mx_b = pt_batch_to_mlx(batch)
            mel, labels = mx_b["mel"], mx_b["labels"]
            B, L = labels.shape

            bos = mx.full((B, 1), _WHISPER_BOS, dtype=mx.int32)
            decoder_ids = mx.concatenate(
                [bos, mx.clip(labels[:, :-1], a_min=0, a_max=None)], axis=1
            )

            # Teacher forward (no grad — outside value_and_grad)
            t_hiddens = get_teacher_hiddens(teacher, mel, decoder_ids)
            mx.eval(*t_hiddens)

            # Student forward + backward
            loss, grads = loss_and_grad_fn(student, mel, decoder_ids, t_hiddens)
            _, grads = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"  [S1] Non-finite loss at step {global_step} — skipping",
                      flush=True)
                global_step += 1
                continue

            losses.append(loss_val)
            writer.add_scalar("stage1/cosine_loss", loss_val, global_step)
            writer.add_scalar("stage1/lr", float(optimizer.learning_rate), global_step)
            global_step += 1

            if global_step % log_every == 0:
                val_str = (f" | val_cos {last_val_loss:.4f}"
                           if math.isfinite(last_val_loss) else "")
                print(f"  [S1] Step {global_step} | cos_loss {loss_val:.4f}{val_str}",
                      flush=True)

            if global_step % eval_every == 0:
                last_val_loss = _val_cosine_loss(
                    student, teacher, val_loader, max_batches=30
                )
                writer.add_scalar("stage1/val_cosine_loss", last_val_loss, global_step)
                writer.flush()
                print(f"  [S1 Val] step {global_step} | val_cos {last_val_loss:.4f}",
                      flush=True)

        ckpt_io.save(
            student,
            f"{checkpoint_dir}/stage1_epoch_{epoch + 1}",
            meta={"epoch": epoch + 1, "step": global_step,
                  "val_cos": last_val_loss},
        )

    writer.close()
    return losses


def _val_cosine_loss(student, teacher, val_loader, max_batches: int = 30) -> float:
    """Cosine distance on validation set — no gradient computation."""
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = pt_batch_to_mlx(batch)
        mel, labels = mx_b["mel"], mx_b["labels"]
        B, L = labels.shape

        bos = mx.full((B, 1), _WHISPER_BOS, dtype=mx.int32)
        decoder_ids = mx.concatenate(
            [bos, mx.clip(labels[:, :-1], a_min=0, a_max=None)], axis=1
        )

        t_hiddens = get_teacher_hiddens(teacher, mel, decoder_ids)
        _, s_hiddens = student(mel, decoder_ids)
        mx.eval(*t_hiddens, *s_hiddens)

        val = cosine_distill_loss(s_hiddens, t_hiddens)
        mx.eval(val)

        v = val.item()
        if math.isfinite(v):
            total += v
            count += 1

    return total / max(count, 1)


# ── Stage 2 ───────────────────────────────────────────────────────────────────

def train_mlx_stage2(
    student,
    train_loader,
    val_loader,
    processor=None,
    epochs: int = 5,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    checkpoint_dir: str = "./checkpoints/mlx",
    tb_log_dir: str = "runs/",
    log_every: int = 50,
    eval_every: int = 500,
    wer_every: int = 1000,
    ss_max_p: float = 0.5,
) -> list[float]:
    """Stage 2: end-to-end ASR fine-tuning with scheduled sampling.

    Scheduled sampling ramps from 0 → ss_max_p over the first half of training,
    then holds.  Closes the teacher-forcing / autoregressive gap.
    """
    student.freeze_for_stage2()
    print(f"[Stage 2] {student.param_summary()}", flush=True)
    print(
        f"[Stage 2] Scheduled sampling: 0 → {ss_max_p:.0%} over first half of training",
        flush=True,
    )

    run_name = datetime.now().strftime("mlx_stage2_%Y%m%d_%H%M%S")
    log_dir = Path(tb_log_dir) / run_name
    writer = SummaryWriter(log_dir=str(log_dir))
    print(f"[Stage 2] TensorBoard → {log_dir}", flush=True)

    total_steps = epochs * len(train_loader)
    optimizer = optim.Adam(learning_rate=lr)

    def s2_loss_fn(model, mel, decoder_ids, labels):
        logits, _ = model(mel, decoder_ids)
        return ce_loss(logits, labels)

    loss_and_grad_fn = nn.value_and_grad(student, s2_loss_fn)

    losses: list[float] = []
    global_step = 0
    last_val_loss = float("nan")
    last_wer = float("nan")

    for epoch in range(epochs):
        print(f"\n{'='*55}", flush=True)
        print(f"[Stage 2] Epoch {epoch + 1} / {epochs}", flush=True)
        print(f"{'='*55}", flush=True)

        for batch in tqdm(train_loader, desc=f"[S2] Epoch {epoch+1}"):
            optimizer.learning_rate = cosine_lr_with_warmup(
                global_step, lr_max=lr, total_steps=total_steps,
                warmup_steps=warmup_steps,
            )

            mx_b = pt_batch_to_mlx(batch)
            mel, labels = mx_b["mel"], mx_b["labels"]
            B, L = labels.shape

            bos = mx.full((B, 1), _WHISPER_BOS, dtype=mx.int32)
            decoder_ids = mx.concatenate(
                [bos, mx.clip(labels[:, :-1], a_min=0, a_max=None)], axis=1
            )

            # ── Scheduled sampling ────────────────────────────────────────────
            # p ramps 0 → ss_max_p over first 50% of training, then holds.
            current_p = ss_max_p * min(
                1.0, global_step / max(total_steps * 0.5, 1)
            )

            if current_p > 0.0:
                # No-grad forward to get predictions for token mixing
                logits_ss, _ = student(mel, decoder_ids)
                mx.eval(logits_ss)
                pred_tokens = mx.clip(mx.argmax(logits_ss, axis=-1), a_min=0, a_max=None)
                mx.eval(pred_tokens)

                # Sample binary mask: True → use model prediction at that position
                use_pred = mx.random.uniform(shape=(B, L - 1)) < current_p
                decoder_ids = mx.concatenate([
                    decoder_ids[:, :1],
                    mx.where(use_pred, pred_tokens[:, :-1], decoder_ids[:, 1:]),
                ], axis=1)
                mx.eval(decoder_ids)

            # ── Training forward + backward ───────────────────────────────────
            loss, grads = loss_and_grad_fn(student, mel, decoder_ids, labels)
            _, grads = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"  [S2] Non-finite loss at step {global_step} — skipping",
                      flush=True)
                global_step += 1
                continue

            losses.append(loss_val)
            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/lr", float(optimizer.learning_rate), global_step)
            writer.add_scalar("train/ss_p", current_p, global_step)
            global_step += 1

            if global_step % log_every == 0:
                parts = [
                    f"Step {global_step}",
                    f"loss {loss_val:.4f}",
                    f"ss_p {current_p:.2f}",
                ]
                if math.isfinite(last_val_loss):
                    parts.append(f"val_loss {last_val_loss:.4f}")
                if math.isfinite(last_wer):
                    parts.append(f"WER {last_wer * 100:.1f}%")
                print("  [S2] " + " | ".join(parts), flush=True)

            if global_step % eval_every == 0:
                last_val_loss = compute_val_loss_mlx(
                    student, val_loader, bos_id=_WHISPER_BOS, max_batches=30
                )
                writer.add_scalar("val/loss", last_val_loss, global_step)
                writer.flush()
                print(
                    f"  [S2 Val] step {global_step} | val_loss {last_val_loss:.4f}",
                    flush=True,
                )

            if processor is not None and global_step % wer_every == 0:
                wer = compute_wer_mlx(student, val_loader, processor, max_batches=10)
                if wer is not None:
                    last_wer = wer
                    writer.add_scalar("val/wer", wer, global_step)
                    writer.flush()
                    print(
                        f"  [S2 WER] step {global_step} | WER {wer * 100:.1f}%",
                        flush=True,
                    )

        # ── Epoch end ─────────────────────────────────────────────────────────
        ckpt_io.save(
            student,
            f"{checkpoint_dir}/stage2_epoch_{epoch + 1}",
            meta={"epoch": epoch + 1, "step": global_step,
                  "val_loss": last_val_loss, "wer": last_wer},
        )

        if processor is not None:
            wer = compute_wer_mlx(student, val_loader, processor, max_batches=20)
            if wer is not None:
                last_wer = wer
                writer.add_scalar("val/wer", wer, global_step)
                writer.flush()
                print(
                    f"  [S2 Epoch WER] epoch {epoch + 1} | WER {wer * 100:.1f}%",
                    flush=True,
                )

    writer.close()
    return losses
