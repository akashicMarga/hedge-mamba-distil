"""MLX training loops for ParlerTTS-Mamba distillation.

Mirrors src/mlx/trainer.py (Whisper) with two adaptations:

1. Encoder split: T5 is large (24 layers) so enc_hidden is pre-computed
   once per batch outside the grad-fn and shared by teacher + student.
   This halves T5 compute vs running it inside value_and_grad.

2. 9-codebook CE loss: Parler's LM head predicts
   (num_codebooks * lm_vocab_size) logits per position.
   Stage 2 loss averages CE across all 9 codebook slots.

MLX training pattern (identical to trainer.py):
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    loss, grads = loss_and_grad_fn(model, *args)
    optimizer.update(model, grads)
    mx.eval(model.parameters(), optimizer.state, loss)

Batch conversion: DataLoader returns torch tensors; call .numpy() before
    passing to mx.array() — same convention as pt_batch_to_mlx in data.py.
"""
import math
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.mlx.loss import cosine_distill_loss
from src.mlx.utils import cosine_lr_with_warmup, clip_grad_norm
from src.mlx.parler_model import (
    ParlerMambaMLX, get_parler_teacher_hiddens, build_first_emb
)
from src.mlx import checkpoint as ckpt_io


def _to_mlx(batch: dict, keys: list[str]) -> dict[str, mx.array]:
    """Convert selected torch-tensor batch keys to MLX arrays via numpy.

    Mirrors the pt_batch_to_mlx() convention in src/mlx/data.py so that
    mx.array() always receives a numpy array, not a raw torch tensor.
    """
    return {k: mx.array(batch[k].numpy()) for k in keys}


# ---------------------------------------------------------------------------
# 9-codebook cross-entropy loss
# ---------------------------------------------------------------------------


def _parler_ce_loss_mlx(
    logits: mx.array,     # (B, T, num_codebooks * lm_vocab_size)
    labels: mx.array,     # (B, T, num_codebooks)  int32, -100 = ignore
    num_codebooks: int,
    lm_vocab_size: int,
) -> mx.array:
    """Mean CE across all codebook slots, masking -100 padding."""
    B, T, _ = logits.shape
    logits_4d = logits.reshape(B, T, num_codebooks, lm_vocab_size)  # (B,T,9,V)

    total   = mx.array(0.0)
    n_valid = 0
    for k in range(num_codebooks):
        lk = logits_4d[:, :, k, :].reshape(B * T, lm_vocab_size)  # (B*T, V)
        yk = labels[:, :, k].reshape(B * T)                       # (B*T,)

        valid      = (yk >= 0).astype(mx.float32)
        safe_yk    = mx.clip(yk, a_min=0, a_max=lm_vocab_size - 1)
        per_token  = nn.losses.cross_entropy(lk, safe_yk, reduction="none")
        cb_loss    = (per_token * valid).sum() / (valid.sum() + 1e-8)
        total      = total + cb_loss
        n_valid   += 1

    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Stage 1 — cosine distillation
# ---------------------------------------------------------------------------


def train_parler_mlx_stage1(
    student: ParlerMambaMLX,
    teacher,                   # IndicParlerTTS (frozen, original attention)
    train_loader,
    val_loader,
    epochs: int = 3,
    lr: float = 5e-4,
    warmup_steps: int = 200,
    checkpoint_dir: str = "./checkpoints/parler_mlx",
    tb_log_dir: str = "runs/",
    log_every: int = 50,
    eval_every: int = 500,
) -> list[float]:
    """Stage 1: train SSM layers to match teacher decoder block outputs.

    Batch format (from make_parler_loaders):
        description_ids : (B, T_desc)        int32
        attention_mask  : (B, T_desc)        int32
        prompt_ids      : (B, T_prompt)      int32
        audio_tokens    : (B, T_audio, 9)    int32
        labels          : (B, T_audio, 9)    int32   (-100 = pad)
    """
    student.freeze_for_stage1()
    teacher.freeze()
    print(f"[Parler S1] {student.param_summary()}", flush=True)

    run_name = datetime.now().strftime("parler_s1_%Y%m%d_%H%M%S")
    log_dir  = Path(tb_log_dir) / run_name
    writer   = SummaryWriter(log_dir=str(log_dir))
    print(f"[Parler S1] TensorBoard → {log_dir}", flush=True)

    total_steps = epochs * len(train_loader)
    optimizer   = optim.Adam(learning_rate=lr)

    def s1_loss_fn(model, enc_hidden, first_emb, t_hiddens):
        _, s_hiddens = model(enc_hidden, first_emb)
        return cosine_distill_loss(s_hiddens, t_hiddens)

    loss_and_grad_fn = nn.value_and_grad(student, s1_loss_fn)

    losses: list[float] = []
    global_step   = 0
    last_val_loss = float("nan")

    for epoch in range(epochs):
        print(f"\n{'='*55}", flush=True)
        print(f"[Parler S1] Epoch {epoch + 1} / {epochs}", flush=True)
        print(f"{'='*55}", flush=True)

        for batch in tqdm(train_loader, desc=f"[S1] Epoch {epoch+1}"):
            optimizer.learning_rate = cosine_lr_with_warmup(
                global_step, lr_max=lr,
                total_steps=total_steps, warmup_steps=warmup_steps,
            )

            # torch tensors → mlx arrays via .numpy() (matches pt_batch_to_mlx)
            mx_b = _to_mlx(batch, ["description_ids", "prompt_ids", "audio_tokens"])
            description_ids = mx_b["description_ids"]
            prompt_ids      = mx_b["prompt_ids"]
            audio_tokens    = mx_b["audio_tokens"]

            # ── Pre-compute frozen encodings (outside grad-fn) ──────────────
            enc_hidden = teacher.text_encoder(description_ids)
            prompt_emb = student.encode_prompt(prompt_ids)        # frozen
            first_emb  = build_first_emb(
                student.model.decoder, prompt_emb, audio_tokens
            )
            mx.eval(enc_hidden, first_emb)

            # ── Teacher hidden states (no grad) ───────────────────────────
            t_hiddens = get_parler_teacher_hiddens(
                teacher, enc_hidden, first_emb
            )
            mx.eval(*t_hiddens)

            # ── Student forward + backward ───────────────────────────────
            loss, grads = loss_and_grad_fn(
                student, enc_hidden, first_emb, t_hiddens
            )
            _, grads = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"  [S1] Non-finite loss step {global_step} — skipping",
                      flush=True)
                global_step += 1
                continue

            losses.append(loss_val)
            writer.add_scalar("stage1/cosine_loss", loss_val, global_step)
            writer.add_scalar(
                "stage1/lr", float(optimizer.learning_rate), global_step
            )
            global_step += 1

            if global_step % log_every == 0:
                val_str = (
                    f" | val {last_val_loss:.4f}"
                    if math.isfinite(last_val_loss) else ""
                )
                print(
                    f"  [S1] Step {global_step} | cos {loss_val:.4f}{val_str}",
                    flush=True,
                )

            if global_step % eval_every == 0:
                last_val_loss = _val_cosine_loss(
                    student, teacher, val_loader, max_batches=20
                )
                writer.add_scalar(
                    "stage1/val_cosine_loss", last_val_loss, global_step
                )
                writer.flush()
                print(
                    f"  [S1 Val] step {global_step} | val {last_val_loss:.4f}",
                    flush=True,
                )

        ckpt_io.save(
            student,
            f"{checkpoint_dir}/stage1_epoch_{epoch + 1}",
            meta={"epoch": epoch + 1, "step": global_step,
                  "val_cos": last_val_loss},
        )

    writer.close()
    return losses


def _val_cosine_loss(
    student: ParlerMambaMLX,
    teacher,
    val_loader,
    max_batches: int = 20,
) -> float:
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = _to_mlx(batch, ["description_ids", "prompt_ids", "audio_tokens"])
        description_ids = mx_b["description_ids"]
        prompt_ids      = mx_b["prompt_ids"]
        audio_tokens    = mx_b["audio_tokens"]

        enc_hidden = teacher.text_encoder(description_ids)
        prompt_emb = student.encode_prompt(prompt_ids)
        first_emb  = build_first_emb(
            student.model.decoder, prompt_emb, audio_tokens
        )
        mx.eval(enc_hidden, first_emb)

        t_hiddens        = get_parler_teacher_hiddens(teacher, enc_hidden, first_emb)
        _, s_hiddens     = student(enc_hidden, first_emb)
        mx.eval(*t_hiddens, *s_hiddens)

        val = cosine_distill_loss(s_hiddens, t_hiddens)
        mx.eval(val)
        v = val.item()
        if math.isfinite(v):
            total += v
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Stage 2 — 9-codebook CE fine-tuning + scheduled sampling
# ---------------------------------------------------------------------------


def train_parler_mlx_stage2(
    student: ParlerMambaMLX,
    train_loader,
    val_loader,
    epochs: int = 5,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    checkpoint_dir: str = "./checkpoints/parler_mlx",
    tb_log_dir: str = "runs/",
    log_every: int = 50,
    eval_every: int = 500,
    ss_max_p: float = 0.5,
) -> list[float]:
    """Stage 2: CE on 9 codebooks with scheduled sampling.

    Scheduled sampling ramps 0 → ss_max_p over the first 50% of training
    (same schedule as Whisper Stage 2 in trainer.py).
    """
    student.freeze_for_stage2()
    print(f"[Parler S2] {student.param_summary()}", flush=True)
    print(
        f"[Parler S2] Scheduled sampling 0 → {ss_max_p:.0%} over first half",
        flush=True,
    )

    run_name = datetime.now().strftime("parler_s2_%Y%m%d_%H%M%S")
    log_dir  = Path(tb_log_dir) / run_name
    writer   = SummaryWriter(log_dir=str(log_dir))
    print(f"[Parler S2] TensorBoard → {log_dir}", flush=True)

    total_steps = epochs * len(train_loader)
    optimizer   = optim.Adam(learning_rate=lr)
    cfg         = student.model.cfg.decoder
    num_cb      = cfg.num_codebooks    # 9
    vocab       = cfg.lm_vocab_size    # 1088

    def s2_loss_fn(model, enc_hidden, first_emb, labels):
        logits, _ = model(enc_hidden, first_emb)
        # Trim logits to T_audio (first_emb includes prompt prefix)
        T_audio = labels.shape[1]
        logits  = logits[:, -T_audio:, :]   # (B, T_audio, 9*V)
        return _parler_ce_loss_mlx(logits, labels, num_cb, vocab)

    loss_and_grad_fn = nn.value_and_grad(student, s2_loss_fn)

    losses: list[float] = []
    global_step   = 0
    last_val_loss = float("nan")

    for epoch in range(epochs):
        print(f"\n{'='*55}", flush=True)
        print(f"[Parler S2] Epoch {epoch + 1} / {epochs}", flush=True)
        print(f"{'='*55}", flush=True)

        for batch in tqdm(train_loader, desc=f"[S2] Epoch {epoch+1}"):
            optimizer.learning_rate = cosine_lr_with_warmup(
                global_step, lr_max=lr,
                total_steps=total_steps, warmup_steps=warmup_steps,
            )

            # torch tensors → mlx arrays via .numpy()
            mx_b = _to_mlx(batch, ["description_ids", "prompt_ids", "audio_tokens", "labels"])
            description_ids = mx_b["description_ids"]
            prompt_ids      = mx_b["prompt_ids"]
            audio_tokens    = mx_b["audio_tokens"]
            labels          = mx_b["labels"]      # (B, T, 9)

            # Frozen encodings outside grad-fn
            enc_hidden = student.encode_description(description_ids)
            prompt_emb = student.encode_prompt(prompt_ids)
            first_emb  = build_first_emb(
                student.model.decoder, prompt_emb, audio_tokens
            )
            mx.eval(enc_hidden, first_emb)

            # ── Scheduled sampling (α la trainer.py Stage 2) ───────────────
            current_p = ss_max_p * min(
                1.0, global_step / max(total_steps * 0.5, 1)
            )
            if current_p > 0.0:
                # No-grad pass to get student predictions
                logits_ss, _ = student(enc_hidden, first_emb)
                mx.eval(logits_ss)
                T_audio   = audio_tokens.shape[1]
                logits_ss = logits_ss[:, -T_audio:, :]     # (B,T,9*V)
                B_, T_, _  = logits_ss.shape
                pred_4d    = logits_ss.reshape(B_, T_, num_cb, vocab).argmax(-1)
                                                            # (B,T,9)
                # Mix: Bernoulli mask per (B, T, codebook)
                swap_mask    = mx.random.uniform(shape=(B_, T_, num_cb)) < current_p
                audio_tokens = mx.where(swap_mask, pred_4d, audio_tokens)
                # Rebuild first_emb with mixed tokens
                first_emb    = build_first_emb(
                    student.model.decoder, prompt_emb, audio_tokens
                )
                mx.eval(first_emb)

            # ── Forward + backward ───────────────────────────────────────
            loss, grads = loss_and_grad_fn(
                student, enc_hidden, first_emb, labels
            )
            _, grads = clip_grad_norm(grads, max_norm=1.0)
            optimizer.update(student, grads)
            mx.eval(student.parameters(), optimizer.state, loss)

            loss_val = loss.item()
            if not math.isfinite(loss_val):
                print(f"  [S2] Non-finite loss step {global_step} — skipping",
                      flush=True)
                global_step += 1
                continue

            losses.append(loss_val)
            writer.add_scalar("stage2/ce_loss",   loss_val,   global_step)
            writer.add_scalar("stage2/ss_p",       current_p,  global_step)
            writer.add_scalar(
                "stage2/lr", float(optimizer.learning_rate), global_step
            )
            global_step += 1

            if global_step % log_every == 0:
                val_str = (
                    f" | val {last_val_loss:.4f}"
                    if math.isfinite(last_val_loss) else ""
                )
                print(
                    f"  [S2] Step {global_step} | ce {loss_val:.4f} | "
                    f"ss_p {current_p:.3f}{val_str}",
                    flush=True,
                )

            if global_step % eval_every == 0:
                last_val_loss = _val_ce_loss(
                    student, val_loader, num_cb, vocab, max_batches=20
                )
                writer.add_scalar("stage2/val_ce_loss", last_val_loss, global_step)
                writer.flush()
                print(
                    f"  [S2 Val] step {global_step} | val {last_val_loss:.4f}",
                    flush=True,
                )

        ckpt_io.save(
            student,
            f"{checkpoint_dir}/stage2_epoch_{epoch + 1}",
            meta={"epoch": epoch + 1, "step": global_step,
                  "val_ce": last_val_loss},
        )

    writer.close()
    return losses


def _val_ce_loss(
    student: ParlerMambaMLX,
    val_loader,
    num_cb: int,
    vocab: int,
    max_batches: int = 20,
) -> float:
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = _to_mlx(batch, ["description_ids", "prompt_ids", "audio_tokens", "labels"])
        description_ids = mx_b["description_ids"]
        prompt_ids      = mx_b["prompt_ids"]
        audio_tokens    = mx_b["audio_tokens"]
        labels          = mx_b["labels"]

        enc_hidden = student.encode_description(description_ids)
        prompt_emb = student.encode_prompt(prompt_ids)
        first_emb  = build_first_emb(
            student.model.decoder, prompt_emb, audio_tokens
        )
        mx.eval(enc_hidden, first_emb)

        logits, _ = student(enc_hidden, first_emb)
        mx.eval(logits)

        T_audio = labels.shape[1]
        logits  = logits[:, -T_audio:, :]
        loss    = _parler_ce_loss_mlx(logits, labels, num_cb, vocab)
        mx.eval(loss)
        v = loss.item()
        if math.isfinite(v):
            total += v
            count += 1
    return total / max(count, 1)
