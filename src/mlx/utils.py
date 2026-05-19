"""Training utilities: LR schedule, grad clipping, WER for MLX."""
import math
import numpy as np
import mlx.core as mx
from mlx.utils import tree_flatten, tree_map


def cosine_lr_with_warmup(
    step: int,
    lr_max: float,
    lr_min: float = 0.0,
    total_steps: int = 1000,
    warmup_steps: int = 200,
) -> float:
    """Linear warmup then cosine decay."""
    if step < warmup_steps:
        return lr_max * max(step, 1) / max(warmup_steps, 1)
    t = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t))


def clip_grad_norm(grads: dict, max_norm: float = 1.0) -> tuple[mx.array, dict]:
    """Clip gradient norm in-place; return (total_norm, clipped_grads)."""
    flat = [v for _, v in tree_flatten(grads) if isinstance(v, mx.array)]
    if not flat:
        return mx.array(0.0), grads
    total_norm = mx.sqrt(sum(mx.sum(g * g) for g in flat))
    scale = mx.minimum(mx.array(1.0), max_norm / (total_norm + 1e-6))
    clipped = tree_map(
        lambda g: g * scale if isinstance(g, mx.array) else g, grads
    )
    return total_norm, clipped


def compute_val_loss_mlx(student, val_loader, bos_id: int, max_batches: int = 30) -> float:
    """Teacher-forced cross-entropy on the validation set (no gradient)."""
    from src.mlx.loss import ce_loss
    from src.mlx.data import pt_batch_to_mlx

    total, count = 0.0, 0
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = pt_batch_to_mlx(batch)
        mel, labels = mx_b["mel"], mx_b["labels"]
        B, L = labels.shape

        bos = mx.full((B, 1), bos_id, dtype=mx.int32)
        dec_ids = mx.concatenate([bos, mx.clip(labels[:, :-1], a_min=0, a_max=None)], axis=1)

        logits, _ = student(mel, dec_ids)
        mx.eval(logits)

        loss_val = ce_loss(logits, labels)
        mx.eval(loss_val)

        v = loss_val.item()
        if math.isfinite(v):
            total += v
            count += 1

    return total / max(count, 1)


def compute_wer_mlx(
    student,
    val_loader,
    processor,
    max_batches: int = 10,
) -> float | None:
    """WER via autoregressive decoding.  Uses mlx_whisper.decoding on student.model."""
    try:
        import jiwer
        import mlx_whisper.decoding as dec
        from src.mlx.data import pt_batch_to_mlx
    except ImportError:
        return None

    gen_opts = dec.DecodingOptions(
        language="en", task="transcribe",
        fp16=False, without_timestamps=True, suppress_blank=True,
    )

    all_preds, all_refs = [], []
    for i, batch in enumerate(val_loader):
        if i >= max_batches:
            break
        mx_b = pt_batch_to_mlx(batch)
        mel = mx_b["mel"]

        labels = batch["labels"]
        refs = processor.batch_decode(labels.clamp(min=0), skip_special_tokens=True)

        for b in range(mel.shape[0]):
            feats = mel[b : b + 1]
            result = dec.decode(student.model, feats, gen_opts)
            all_preds.append(result[0].text.strip())

        all_refs.extend(refs)

    if not all_refs:
        return None
    return jiwer.wer(all_refs, all_preds)
