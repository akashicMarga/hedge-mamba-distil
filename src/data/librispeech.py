"""LibriSpeech dataloader for WhisperMamba training.

Uses HuggingFace datasets + WhisperProcessor.
Input features (mel spectrogram) are fixed shape (80 × 3000) — no padding needed.
Only labels (transcript tokens) vary in length — padded with -100 (ignored in loss).
"""
import torch
import soundfile as sf
import io
import numpy as np
from torch.utils.data import DataLoader
from transformers import WhisperProcessor
from datasets import load_dataset, Audio


class LibriSpeechCollator:
    """Collate variable-length label sequences with -100 padding.
    Defined as a class (not a closure) so multiprocessing can pickle it.
    """

    def __call__(self, batch):
        input_features = torch.stack([
            torch.tensor(b["input_features"], dtype=torch.float32) for b in batch
        ])
        label_lists = [b["labels"] for b in batch]
        max_len = max(len(l) for l in label_lists)
        labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for i, l in enumerate(label_lists):
            labels[i, :len(l)] = torch.tensor(l, dtype=torch.long)
        return {"input_features": input_features, "labels": labels}


def make_librispeech_loaders(
    model_id: str = "openai/whisper-tiny",
    language: str = "en",
    task: str = "transcribe",
    train_split: str = "train.100",
    val_split: str = "validation.clean",
    batch_size: int = 8,
    num_workers: int = 2,
    max_label_length: int = 448,
    cache_dir: str | None = None,
):
    """Return (train_loader, val_loader, processor).

    First call downloads LibriSpeech (~6GB for train.100) and caches it.
    Subsequent calls are instant.
    """
    processor = WhisperProcessor.from_pretrained(
        model_id, language=language, task=task
    )

    def preprocess(example):
        # Decode audio with soundfile directly (bypasses torchcodec dependency).
        # datasets stores raw audio bytes in example["audio"]["bytes"] when
        # decode=False; if a path is available we use that instead.
        audio_info = example["audio"]
        if audio_info.get("bytes"):
            array, sr = sf.read(io.BytesIO(audio_info["bytes"]))
        else:
            array, sr = sf.read(audio_info["path"])
        # Ensure mono float32
        if array.ndim > 1:
            array = array.mean(axis=1)
        array = array.astype(np.float32)

        # Mel spectrogram — always (80, 3000) after Whisper padding
        feats = processor.feature_extractor(
            array,
            sampling_rate=sr,
            return_tensors="np",
        ).input_features[0]  # numpy (80, 3000)

        # Transcript tokens
        labels = processor.tokenizer(
            example["text"].lower(),
            max_length=max_label_length,
            truncation=True,
        ).input_ids

        return {"input_features": feats, "labels": labels}

    print("Loading LibriSpeech train...", flush=True)
    train_ds = load_dataset(
        "librispeech_asr", "clean", split=train_split, cache_dir=cache_dir
    )
    print("Loading LibriSpeech val...", flush=True)
    val_ds = load_dataset(
        "librispeech_asr", "clean", split=val_split, cache_dir=cache_dir
    )

    # Disable datasets' built-in audio decoder — we decode with soundfile ourselves
    train_ds = train_ds.cast_column("audio", Audio(decode=False))
    val_ds = val_ds.cast_column("audio", Audio(decode=False))

    print("Preprocessing train...", flush=True)
    train_ds = train_ds.map(
        preprocess,
        remove_columns=train_ds.column_names,
        num_proc=num_workers,
        desc="Train preprocess",
    )
    print("Preprocessing val...", flush=True)
    val_ds = val_ds.map(
        preprocess,
        remove_columns=val_ds.column_names,
        num_proc=num_workers,
        desc="Val preprocess",
    )

    collate_fn = LibriSpeechCollator()

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=False,
    )

    print(
        f"Dataset ready — train: {len(train_ds):,} samples | val: {len(val_ds):,} samples",
        flush=True,
    )
    return train_loader, val_loader, processor
