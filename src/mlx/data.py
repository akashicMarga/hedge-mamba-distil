"""LibriSpeech data loader for MLX training.

Wraps src.data.librispeech: HF DataLoader → mx.array at batch boundary only.
num_workers=0 is required — MLX arrays cannot cross process boundaries.

Mel format conversion:
  PyTorch: (B, n_mels=80, T=3000)
  MLX:     (B, T=3000, n_mels=80)   ← transpose axes 1 and 2
"""
import numpy as np
import mlx.core as mx
from torch.utils.data import DataLoader

from src.data.librispeech import make_librispeech_loaders, LibriSpeechCollator


def make_mlx_loaders(
    model_id: str = "openai/whisper-tiny",
    language: str = "en",
    task: str = "transcribe",
    train_split: str = "train.100",
    val_split: str = "validation",
    batch_size: int = 8,
    max_label_length: int = 448,
    debug: bool = False,
):
    """Return (train_loader, val_loader, processor).

    Batches are plain PyTorch tensors — call pt_batch_to_mlx() inside the
    training loop to convert at the batch boundary.
    """
    train_loader, val_loader, processor = make_librispeech_loaders(
        model_id=model_id,
        language=language,
        task=task,
        train_split=train_split,
        val_split=val_split,
        batch_size=batch_size,
        num_workers=0,  # MLX arrays cannot cross process boundaries
        max_label_length=max_label_length,
    )

    if debug:
        from torch.utils.data import Subset
        collate_fn = LibriSpeechCollator()
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(100)),
            batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(20)),
            batch_size=batch_size, shuffle=False,
            collate_fn=collate_fn, num_workers=0,
        )
        print("DEBUG mode: 100 train / 20 val samples", flush=True)

    return train_loader, val_loader, processor


def pt_batch_to_mlx(batch: dict) -> dict:
    """Convert a PyTorch DataLoader batch to MLX arrays.

    Returns:
        mel:    (B, T=3000, 80) float32
        labels: (B, L)          int32   — -100 for padding
    """
    # input_features: (B, 80, T=3000) → (B, T=3000, 80)
    mel = mx.array(
        batch["input_features"].numpy().transpose(0, 2, 1), dtype=mx.float32
    )
    labels = mx.array(batch["labels"].numpy().astype(np.int32))
    return {"mel": mel, "labels": labels}
