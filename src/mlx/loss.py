"""Loss functions for MLX training."""
import mlx.core as mx
import mlx.nn as nn


def cosine_distill_loss(
    student_hiddens: list[mx.array],
    teacher_hiddens: list[mx.array],
) -> mx.array:
    """Mean cosine distance across decoder blocks.

    student_hiddens, teacher_hiddens: list of (B, L, D), one per block.
    Loss = mean over blocks of (1 - cosine_similarity).
    """
    total = mx.array(0.0)
    for s, t in zip(student_hiddens, teacher_hiddens):
        s_norm = s / (mx.sqrt((s * s).sum(axis=-1, keepdims=True)) + 1e-8)
        t_norm = t / (mx.sqrt((t * t).sum(axis=-1, keepdims=True)) + 1e-8)
        cos = (s_norm * t_norm).sum(axis=-1)  # (B, L)
        total = total + (1.0 - cos).mean()
    return total / max(len(student_hiddens), 1)


def ce_loss(logits: mx.array, labels: mx.array) -> mx.array:
    """Cross-entropy loss with -100 ignore mask.

    logits: (B, L, vocab)
    labels: (B, L) — -100 for padding positions
    """
    B, L, V = logits.shape
    logits_flat = logits.reshape(-1, V)   # (B*L, vocab)
    labels_flat = labels.reshape(-1)      # (B*L,)

    # Weight: 1 where label is valid, 0 where padding (-100)
    valid = (labels_flat >= 0).astype(mx.float32)          # (B*L,)
    safe_labels = mx.clip(labels_flat, a_min=0, a_max=V - 1)  # avoid OOB on -100

    per_token = nn.losses.cross_entropy(logits_flat, safe_labels, reduction="none")
    return (per_token * valid).sum() / (valid.sum() + 1e-8)
