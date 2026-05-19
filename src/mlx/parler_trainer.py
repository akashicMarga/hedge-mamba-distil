"""MLX training loops for ParlerTTS-Mamba distillation.

Memory-optimised rewrite based on mlx-audio-train/train/trainer.py:

1.  stop_gradient between decoder layers — the SSM scan stores (D, Ns) state
    for ALL L steps per layer during backward. 24 layers × 271 steps × 1024D
    × 128Ns × 4B ≈ 3.4 GB. With stop_gradient it drops to ~140 MB.

2.  _strip_empty — removes empty-dict subtrees (from DACQuantizer's raw list
    arrays) before model.update(), fixing the "invalid type: dict" crash
    without needing to pop 'dac' from the model.

3.  Flat params via tree_flatten / tree_unflatten — more robust than
    nn.value_and_grad(student, ...) which uses model.trainable_parameters().

4.  Gradient accumulation — run micro-batch=1, accumulate N steps so the
    effective batch is large enough without OOM.
"""
import math
import sys
from datetime import datetime
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mxu
import numpy as np
from tqdm import tqdm

from src.mlx.loss import cosine_distill_loss
from src.mlx.parler_model import (
    ParlerMambaMLX, build_first_emb, _causal_mask
)
from src.mlx import checkpoint as ckpt_io


# ---------------------------------------------------------------------------
# Batch conversion
# ---------------------------------------------------------------------------

def _to_mlx(batch: dict, keys: list[str]) -> dict[str, mx.array]:
    return {k: mx.array(batch[k].numpy()) for k in keys}


# ---------------------------------------------------------------------------
# _strip_empty (from mlx-audio-train/train/trainer.py)
# Removes empty-dict subtrees so model.update() never sees a dict where an
# array is expected (DACQuantizer stores weights in raw Python lists).
# ---------------------------------------------------------------------------

def _is_empty(x) -> bool:
    if isinstance(x, dict):
        return all(_is_empty(v) for v in x.values()) if x else True
    if isinstance(x, list):
        return all(_is_empty(v) for v in x) if x else True
    return False


def _strip_empty(d):
    if isinstance(d, dict):
        out = {}
        for k, v in d.items():
            s = _strip_empty(v)
            if not _is_empty(s):
                out[k] = s
        return out
    if isinstance(d, list):
        out = [_strip_empty(v) for v in d]
        return [v for v in out if not _is_empty(v)]
    return d


# ---------------------------------------------------------------------------
# Gradient helpers (from mlx-audio-train)
# ---------------------------------------------------------------------------

def _add_grads(a, b):
    if isinstance(a, mx.array) and isinstance(b, mx.array):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        out = {}
        for k in a:
            if k in b:
                out[k] = _add_grads(a[k], b[k])
            else:
                out[k] = a[k]
        for k in b:
            if k not in out:
                out[k] = b[k]
        return out
    if isinstance(a, list) and isinstance(b, list):
        return [_add_grads(x, y) for x, y in zip(a, b)]
    return a


def _clip_grads(flat_grads: dict, max_norm: float) -> dict:
    if max_norm <= 0:
        return flat_grads
    sq = [mx.sum(g ** 2) for g in flat_grads.values() if isinstance(g, mx.array)]
    if not sq:
        return flat_grads
    total_sq = mx.sum(mx.stack(sq))
    mx.eval(total_sq)
    norm = math.sqrt(float(total_sq))
    if norm > max_norm:
        scale = max_norm / (norm + 1e-8)
        return {k: v * scale if isinstance(v, mx.array) else v
                for k, v in flat_grads.items()}
    return flat_grads


# ---------------------------------------------------------------------------
# 9-codebook cross-entropy loss
# ---------------------------------------------------------------------------

def _parler_ce_loss_mlx(
    logits: mx.array,
    labels: mx.array,
    num_codebooks: int,
    lm_vocab_size: int,
) -> mx.array:
    B, T, _ = logits.shape
    logits_4d = logits.reshape(B, T, num_codebooks, lm_vocab_size)
    total, n_valid = mx.array(0.0), 0
    for k in range(num_codebooks):
        lk     = logits_4d[:, :, k, :].reshape(B * T, lm_vocab_size)
        yk     = labels[:, :, k].reshape(B * T)
        valid  = (yk >= 0).astype(mx.float32)
        safe_y = mx.clip(yk, a_min=0, a_max=lm_vocab_size - 1)
        per_t  = nn.losses.cross_entropy(lk, safe_y, reduction="none")
        total  = total + (per_t * valid).sum() / (valid.sum() + 1e-8)
        n_valid += 1
    return total / max(n_valid, 1)


# ---------------------------------------------------------------------------
# Stage 1 loss with stop_gradient between layers
# ---------------------------------------------------------------------------

def _s1_loss_layerwise(
    student: ParlerMambaMLX,
    enc_hidden: mx.array,
    first_emb: mx.array,
    t_hiddens: list[mx.array],
) -> mx.array:
    """Cosine distillation loss with stop_gradient between decoder layers.

    stop_gradient on each layer's INPUT limits the SSM backward graph to
    ONE layer at a time: O(L*D*Ns) instead of O(depth*L*D*Ns).
    Reduces backward activation memory from ~3.4 GB to ~140 MB.
    """
    T    = first_emb.shape[1]
    mask = _causal_mask(T, dtype=first_emb.dtype)

    x     = first_emb
    total = mx.array(0.0)
    n     = len(t_hiddens)

    for i, layer in enumerate(student.model.decoder.layers):
        # Stop gradient so the backward of layer i doesn't flow into layer i-1
        x_in = mx.stop_gradient(x)
        x    = layer(x_in, enc_hidden, mask=mask, self_cache=None, cross_cache=None)

        if i < n:
            t = t_hiddens[i]
            # cosine similarity: (B, T) → mean
            num  = (x.astype(mx.float32) * t.astype(mx.float32)).sum(axis=-1)
            dnorm = mx.sqrt(
                (x.astype(mx.float32) ** 2).sum(axis=-1) *
                (t.astype(mx.float32) ** 2).sum(axis=-1) + 1e-8
            )
            sim   = (num / dnorm).mean()
            total = total + (1.0 - sim)

    return total / max(n, 1)


# ---------------------------------------------------------------------------
# Teacher hidden states (no grad, computed outside value_and_grad)
# ---------------------------------------------------------------------------

def _get_teacher_hiddens(
    teacher,
    enc_hidden: mx.array,
    first_emb: mx.array,
) -> list[mx.array]:
    T    = first_emb.shape[1]
    mask = _causal_mask(T, dtype=first_emb.dtype)
    x    = first_emb
    hs   = []
    for layer in teacher.decoder.layers:
        x = layer(x, enc_hidden, mask=mask, self_cache=None, cross_cache=None)
        hs.append(x)
    return hs


# ---------------------------------------------------------------------------
# Stage 1 — cosine distillation
# ---------------------------------------------------------------------------

def train_parler_mlx_stage1(
    student: ParlerMambaMLX,
    teacher,
    train_loader,
    val_loader,
    epochs: int = 3,
    lr: float = 5e-4,
    warmup_steps: int = 200,
    grad_accumulation: int = 4,
    checkpoint_dir: str = "./checkpoints/parler_mlx",
    tb_log_dir: str = "runs/",
    log_every: int = 50,
    eval_every: int = 500,
) -> list[float]:
    """Stage 1: cosine distillation with stop_gradient between layers.

    Uses gradient accumulation (default 4) so micro-batch=1 is safe even
    on 16 GB unified-memory Macs.
    """
    student.freeze_for_stage1()
    teacher.freeze()
    print(f"[Parler S1] {student.param_summary()}", flush=True)
    print(f"[Parler S1] grad_accumulation={grad_accumulation}", flush=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = datetime.now().strftime("parler_s1_%Y%m%d_%H%M%S")
        log_dir  = Path(tb_log_dir) / run_name
        writer   = SummaryWriter(log_dir=str(log_dir))
        print(f"[Parler S1] TensorBoard → {log_dir}", flush=True)
    except ImportError:
        writer = None

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    total_steps = epochs * (len(train_loader) // grad_accumulation)

    # Build value_and_grad using flat trainable params + _strip_empty
    # (avoids DACQuantizer empty-dict crash in model.update())
    def _inner(params, enc_h, first_e, t_hs):
        nested = mxu.tree_unflatten(list(params.items()))
        student.update(nested)
        return _s1_loss_layerwise(student, enc_h, first_e, t_hs)

    _raw_vg = mx.value_and_grad(_inner)

    def _step_vg(enc_h, first_e, t_hs):
        flat = dict(mxu.tree_flatten(student.trainable_parameters()))
        loss, grads = _raw_vg(flat, enc_h, first_e, t_hs)
        mx.eval(loss, grads)
        return loss, _strip_empty(grads)

    optimizer = optim.Adam(learning_rate=lr)
    state     = [student.state, optimizer.state]

    losses: list[float] = []
    global_step   = 0
    accum_grads   = {}
    accum_loss    = 0.0
    accum_count   = 0
    last_val_loss = float("nan")

    for epoch in range(epochs):
        print(f"\n{'='*55}", flush=True)
        print(f"[Parler S1] Epoch {epoch + 1} / {epochs}", flush=True)

        for batch in tqdm(train_loader, desc=f"[S1] E{epoch+1}"):
            lr_now = _cosine_lr(global_step * grad_accumulation,
                                lr, total_steps * grad_accumulation, warmup_steps)
            optimizer.learning_rate = lr_now

            mx_b = _to_mlx(batch, ["description_ids", "attention_mask", "prompt_ids", "audio_tokens"])

            # Frozen encodings outside grad-fn
            enc_hidden = student.encode_description(
                mx_b["description_ids"], mx_b["attention_mask"]
            )
            prompt_emb = student.encode_prompt(mx_b["prompt_ids"])
            first_emb  = build_first_emb(
                student.model.decoder, prompt_emb, mx_b["audio_tokens"]
            )
            mx.eval(enc_hidden, first_emb)

            # Teacher hiddens (no grad, materialized before student backward)
            t_hiddens = _get_teacher_hiddens(teacher, enc_hidden, first_emb)
            mx.eval(*t_hiddens)

            # Gradient accumulation micro-step
            loss, grads = _step_vg(enc_hidden, first_emb, t_hiddens)

            loss_val = float(loss)
            if not math.isfinite(loss_val):
                print(f"  [S1] Non-finite loss — skipping", flush=True)
                accum_count += 1
                if accum_count >= grad_accumulation:
                    accum_grads, accum_loss, accum_count = {}, 0.0, 0
                    global_step += 1
                continue

            accum_grads = grads if not accum_grads else _add_grads(accum_grads, grads)
            accum_loss += loss_val
            accum_count += 1

            if accum_count < grad_accumulation:
                continue

            # Optimizer step
            flat_g = dict(mxu.tree_flatten(accum_grads))
            flat_g = _clip_grads(flat_g, max_norm=1.0)
            optimizer.update(student, mxu.tree_unflatten(list(flat_g.items())))
            mx.eval(state)

            step_loss = accum_loss / accum_count
            losses.append(step_loss)
            accum_grads, accum_loss, accum_count = {}, 0.0, 0

            if writer:
                writer.add_scalar("stage1/cosine_loss", step_loss, global_step)
                writer.add_scalar("stage1/lr", float(optimizer.learning_rate), global_step)

            if global_step % log_every == 0:
                val_s = f" | val {last_val_loss:.4f}" if math.isfinite(last_val_loss) else ""
                print(f"  [S1] step {global_step} | cos {step_loss:.4f}{val_s}", flush=True)

            if global_step % eval_every == 0 and global_step > 0:
                last_val_loss = _val_cosine(student, teacher, val_loader, max_batches=10)
                if writer:
                    writer.add_scalar("stage1/val_cosine_loss", last_val_loss, global_step)
                    writer.flush()
                print(f"  [S1 Val] step {global_step} | val {last_val_loss:.4f}", flush=True)

            global_step += 1

        ckpt_io.save(
            student,
            f"{checkpoint_dir}/stage1_epoch_{epoch + 1}",
            meta={"epoch": epoch + 1, "step": global_step, "val_cos": last_val_loss},
        )
        print(f"[Parler S1] Checkpoint → {checkpoint_dir}/stage1_epoch_{epoch+1}", flush=True)

    if writer:
        writer.close()
    return losses


def _val_cosine(student, teacher, val_loader, max_batches=10) -> float:
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = _to_mlx(batch, ["description_ids", "attention_mask", "prompt_ids", "audio_tokens"])
        enc_hidden = student.encode_description(mx_b["description_ids"], mx_b["attention_mask"])
        prompt_emb = student.encode_prompt(mx_b["prompt_ids"])
        first_emb  = build_first_emb(student.model.decoder, prompt_emb, mx_b["audio_tokens"])
        mx.eval(enc_hidden, first_emb)
        t_hs = _get_teacher_hiddens(teacher, enc_hidden, first_emb)
        mx.eval(*t_hs)
        v = _s1_loss_layerwise(student, enc_hidden, first_emb, t_hs)
        mx.eval(v)
        val = float(v)
        if math.isfinite(val):
            total += val
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# Stage 2 — 9-codebook CE + scheduled sampling
# ---------------------------------------------------------------------------

def train_parler_mlx_stage2(
    student: ParlerMambaMLX,
    train_loader,
    val_loader,
    epochs: int = 5,
    lr: float = 1e-4,
    warmup_steps: int = 500,
    grad_accumulation: int = 4,
    checkpoint_dir: str = "./checkpoints/parler_mlx",
    tb_log_dir: str = "runs/",
    log_every: int = 50,
    eval_every: int = 500,
    ss_max_p: float = 0.5,
) -> list[float]:
    student.freeze_for_stage2()
    print(f"[Parler S2] {student.param_summary()}", flush=True)
    print(f"[Parler S2] grad_accumulation={grad_accumulation}", flush=True)

    try:
        from torch.utils.tensorboard import SummaryWriter
        run_name = datetime.now().strftime("parler_s2_%Y%m%d_%H%M%S")
        log_dir  = Path(tb_log_dir) / run_name
        writer   = SummaryWriter(log_dir=str(log_dir))
        print(f"[Parler S2] TensorBoard → {log_dir}", flush=True)
    except ImportError:
        writer = None

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    total_steps = epochs * (len(train_loader) // grad_accumulation)

    cfg    = student.model.cfg.decoder
    num_cb = cfg.num_codebooks
    vocab  = cfg.lm_vocab_size

    def _s2_inner(params, enc_h, first_e, labels):
        nested = mxu.tree_unflatten(list(params.items()))
        student.update(nested)
        logits, _ = student(enc_h, first_e)
        T_audio    = labels.shape[1]
        logits     = logits[:, -T_audio:, :]
        return _parler_ce_loss_mlx(logits, labels, num_cb, vocab)

    _s2_raw_vg = mx.value_and_grad(_s2_inner)

    def _s2_step_vg(enc_h, first_e, labels):
        flat = dict(mxu.tree_flatten(student.trainable_parameters()))
        loss, grads = _s2_raw_vg(flat, enc_h, first_e, labels)
        mx.eval(loss, grads)
        return loss, _strip_empty(grads)

    optimizer = optim.Adam(learning_rate=lr)
    state     = [student.state, optimizer.state]

    losses: list[float] = []
    global_step   = 0
    accum_grads   = {}
    accum_loss    = 0.0
    accum_count   = 0
    last_val_loss = float("nan")

    for epoch in range(epochs):
        print(f"\n{'='*55}", flush=True)
        print(f"[Parler S2] Epoch {epoch + 1} / {epochs}", flush=True)

        for batch in tqdm(train_loader, desc=f"[S2] E{epoch+1}"):
            lr_now = _cosine_lr(global_step * grad_accumulation,
                                lr, total_steps * grad_accumulation, warmup_steps)
            optimizer.learning_rate = lr_now

            mx_b = _to_mlx(batch, ["description_ids", "attention_mask", "prompt_ids", "audio_tokens", "labels"])
            description_ids = mx_b["description_ids"]
            attention_mask  = mx_b["attention_mask"]
            prompt_ids      = mx_b["prompt_ids"]
            audio_tokens    = mx_b["audio_tokens"]
            labels          = mx_b["labels"]

            enc_hidden = student.encode_description(description_ids, attention_mask)
            prompt_emb = student.encode_prompt(prompt_ids)
            first_emb  = build_first_emb(student.model.decoder, prompt_emb, audio_tokens)
            mx.eval(enc_hidden, first_emb)

            # Scheduled sampling
            current_p = ss_max_p * min(1.0, (global_step * grad_accumulation) / max(total_steps * grad_accumulation * 0.5, 1))
            if current_p > 0.0:
                logits_ss, _ = student(enc_hidden, first_emb)
                mx.eval(logits_ss)
                T_audio   = audio_tokens.shape[1]
                logits_ss = logits_ss[:, -T_audio:, :]
                B_, T_, _ = logits_ss.shape
                pred_4d   = logits_ss.reshape(B_, T_, num_cb, vocab).argmax(-1)
                swap      = mx.random.uniform(shape=(B_, T_, num_cb)) < current_p
                audio_tokens = mx.where(swap, pred_4d, audio_tokens)
                first_emb = build_first_emb(student.model.decoder, prompt_emb, audio_tokens)
                mx.eval(first_emb)

            loss, grads = _s2_step_vg(enc_hidden, first_emb, labels)

            loss_val = float(loss)
            if not math.isfinite(loss_val):
                accum_count += 1
                if accum_count >= grad_accumulation:
                    accum_grads, accum_loss, accum_count = {}, 0.0, 0
                    global_step += 1
                continue

            accum_grads = grads if not accum_grads else _add_grads(accum_grads, grads)
            accum_loss += loss_val
            accum_count += 1

            if accum_count < grad_accumulation:
                continue

            flat_g = dict(mxu.tree_flatten(accum_grads))
            flat_g = _clip_grads(flat_g, max_norm=1.0)
            optimizer.update(student, mxu.tree_unflatten(list(flat_g.items())))
            mx.eval(state)

            step_loss = accum_loss / accum_count
            losses.append(step_loss)
            accum_grads, accum_loss, accum_count = {}, 0.0, 0

            if writer:
                writer.add_scalar("stage2/ce_loss", step_loss, global_step)
                writer.add_scalar("stage2/ss_p", current_p, global_step)
                writer.add_scalar("stage2/lr", float(optimizer.learning_rate), global_step)

            if global_step % log_every == 0:
                val_s = f" | val {last_val_loss:.4f}" if math.isfinite(last_val_loss) else ""
                print(f"  [S2] step {global_step} | ce {step_loss:.4f} | ss_p {current_p:.3f}{val_s}", flush=True)

            if global_step % eval_every == 0 and global_step > 0:
                last_val_loss = _val_ce(student, val_loader, num_cb, vocab, max_batches=10)
                if writer:
                    writer.add_scalar("stage2/val_ce_loss", last_val_loss, global_step)
                    writer.flush()
                print(f"  [S2 Val] step {global_step} | val {last_val_loss:.4f}", flush=True)

            global_step += 1

        ckpt_io.save(
            student,
            f"{checkpoint_dir}/stage2_epoch_{epoch + 1}",
            meta={"epoch": epoch + 1, "step": global_step, "val_ce": last_val_loss},
        )

    if writer:
        writer.close()
    return losses


def _val_ce(student, val_loader, num_cb, vocab, max_batches=10) -> float:
    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = _to_mlx(batch, ["description_ids", "attention_mask", "prompt_ids", "audio_tokens", "labels"])
        enc_hidden = student.encode_description(mx_b["description_ids"], mx_b["attention_mask"])
        prompt_emb = student.encode_prompt(mx_b["prompt_ids"])
        first_emb  = build_first_emb(student.model.decoder, prompt_emb, mx_b["audio_tokens"])
        mx.eval(enc_hidden, first_emb)
        logits, _ = student(enc_hidden, first_emb)
        mx.eval(logits)
        T_audio = mx_b["labels"].shape[1]
        loss = _parler_ce_loss_mlx(logits[:, -T_audio:, :], mx_b["labels"], num_cb, vocab)
        mx.eval(loss)
        v = float(loss)
        if math.isfinite(v):
            total += v
            count += 1
    return total / max(count, 1)


# ---------------------------------------------------------------------------
# LR schedule
# ---------------------------------------------------------------------------

def _cosine_lr(step: int, lr_max: float, total_steps: int, warmup: int) -> float:
    if step < warmup:
        return lr_max * max(step, 1) / max(warmup, 1)
    t = (step - warmup) / max(total_steps - warmup, 1)
    return max(lr_max * 0.5 * (1 + math.cos(math.pi * t)), 1e-7)
