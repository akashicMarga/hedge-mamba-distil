"""Data loading for ParlerTTS-Mamba distillation.

Mirrors src/mlx/data.py (LibriSpeech) but for the Parler TTS task.

Batch format
------------
    description_ids : (B, T_desc)      int32  — T5 tokenized description
    attention_mask  : (B, T_desc)      int32  — T5 attention mask
    prompt_ids      : (B, T_prompt)    int32  — custom tokenized text-to-speak
    audio_tokens    : (B, T_audio, 9)  int32  — DAC codec tokens (delayed)
    labels          : (B, T_audio, 9)  int32  — -100 where padded

Preprocessing pipeline (offline, run once before training)
----------------------------------------------------------
1.  Download parler-tts/parler-tts-mini-v1 from HuggingFace.
2.  For each sample:
    a.  T5-tokenize description → description_ids / attention_mask.
    b.  Custom-tokenize prompt text → prompt_ids.
    c.  Run teacher model's audio encoder (DAC) on the audio waveform
        → codec tokens, shape (9, T_raw).
    d.  Apply delay pattern → decoder_input_ids (9, T_audio),
        labels (9, T_audio) with -100 for padded positions.
3.  Cache to disk as .npz files.

This module provides:
    preprocess_and_cache()  — run offline once
    ParlerDistilDataset    — torch Dataset reading cached .npz files
    make_parler_loaders()  — returns (train_loader, val_loader)
"""
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class ParlerDistilDataset(Dataset):
    """Reads pre-processed .npz files produced by preprocess_and_cache()."""

    def __init__(self, cache_dir: str, max_audio_len: int = 512):
        self.files = sorted(Path(cache_dir).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(
                f"No .npz files found in {cache_dir}.  "
                f"Run preprocess_and_cache() first."
            )
        self.max_audio_len = max_audio_len

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
        data = np.load(self.files[idx])
        T    = min(self.max_audio_len, data["audio_tokens"].shape[0])
        return {
            "description_ids": data["description_ids"].astype(np.int32),
            "attention_mask":  data["attention_mask"].astype(np.int32),
            "prompt_ids":      data["prompt_ids"].astype(np.int32),
            # audio_tokens + labels: (T, 9)
            "audio_tokens":    data["audio_tokens"][:T].astype(np.int32),
            "labels":          data["labels"][:T].astype(np.int32),
        }


def _collate(batch: list[dict]) -> dict:
    """Pad each field to the longest sequence in the batch."""
    def pad_2d(arrays, pad_val=0):
        L = max(a.shape[0] for a in arrays)
        return np.stack([
            np.pad(a, ((0, L - a.shape[0]),) + ((0,0),) * (a.ndim - 1),
                   constant_values=pad_val)
            for a in arrays
        ])

    return {
        "description_ids": torch.from_numpy(pad_2d(
            [s["description_ids"] for s in batch]
        )),
        "attention_mask":  torch.from_numpy(pad_2d(
            [s["attention_mask"] for s in batch]
        )),
        "prompt_ids": torch.from_numpy(pad_2d(
            [s["prompt_ids"] for s in batch]
        )),
        "audio_tokens": torch.from_numpy(pad_2d(
            [s["audio_tokens"] for s in batch]
        )),
        "labels": torch.from_numpy(pad_2d(
            [s["labels"] for s in batch], pad_val=-100
        )),
    }


def make_parler_loaders(
    train_cache_dir: str,
    val_cache_dir: str,
    batch_size: int = 4,
    max_audio_len: int = 512,
) -> tuple[DataLoader, DataLoader]:
    """Return (train_loader, val_loader).

    num_workers=0 required — MLX arrays cannot cross process boundaries.
    """
    train_ds = ParlerDistilDataset(train_cache_dir, max_audio_len)
    val_ds   = ParlerDistilDataset(val_cache_dir,   max_audio_len)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        collate_fn=_collate, num_workers=0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        collate_fn=_collate, num_workers=0,
    )
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# Offline preprocessing
# ---------------------------------------------------------------------------


def preprocess_and_cache(
    hf_dataset_id: str = "parler-tts/parler-tts-mini-v1",
    out_dir: str = "./data/parler_distil",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_audio_len_s: float = 6.0,
    device: str = "cpu",
) -> None:
    """Pre-process a HuggingFace Parler TTS dataset into .npz cache files.

    Requires:
        pip install datasets transformers parler_tts
        mlx-audio-train on sys.path (for IndicParlerTTS + DAC encoder)

    The teacher's audio encoder (DAC) runs on `device`.
    Set device='mps' for Apple Silicon GPU acceleration during preprocessing.
    """
    import sys
    from datasets import load_dataset

    # Lazy imports (mlx-audio-train path must be set by caller)
    try:
        from models.indic_parler_tts.model import IndicParlerTTS
        import mlx.core as mx
    except ImportError as e:
        raise ImportError(
            "mlx-audio-train not found on sys.path. "
            "Add it with: sys.path.insert(0, '/path/to/mlx-audio-train')"
        ) from e

    from parler_tts import ParlerTTSForConditionalGeneration
    from transformers import AutoTokenizer
    import soundfile as sf
    import io

    out_path = Path(out_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"[preprocess] Loading dataset {hf_dataset_id} / {split} ...")
    ds = load_dataset(hf_dataset_id, split=split)
    if max_samples:
        ds = ds.select(range(min(max_samples, len(ds))))

    print("[preprocess] Loading teacher model + tokenizers ...")
    # HF model for T5 tokenizer + DAC encoder (audio → codec tokens)
    hf_model = ParlerTTSForConditionalGeneration.from_pretrained(
        "ai4bharat/indic-parler-tts"
    ).to(device)
    hf_model.eval()

    t5_tokenizer  = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    # Prompt tokenizer is the same T5 tokenizer for IndicParlerTTS
    pmt_tokenizer = t5_tokenizer

    max_audio_frames = int(max_audio_len_s * 44100 / 512)

    print(f"[preprocess] Processing {len(ds)} samples → {out_path}")
    for idx, sample in enumerate(ds):
        # 1. T5-tokenize description
        desc_enc = t5_tokenizer(
            sample["description"], return_tensors="pt",
            truncation=True, max_length=256,
        )
        description_ids = desc_enc["input_ids"][0].numpy()    # (T_desc,)
        attention_mask  = desc_enc["attention_mask"][0].numpy()

        # 2. Tokenize prompt text (text to speak)
        pmt_enc  = pmt_tokenizer(
            sample["text"], return_tensors="pt",
            truncation=True, max_length=128,
        )
        prompt_ids = pmt_enc["input_ids"][0].numpy()          # (T_prompt,)

        # 3. Decode audio to waveform, then encode with DAC
        audio_arr = sample["audio"]["array"]   # float32 waveform
        sr        = sample["audio"]["sampling_rate"]
        import torch as _torch
        with _torch.no_grad():
            # HF audio encoder expects (batch, time)
            wav_t = _torch.tensor(audio_arr, dtype=_torch.float32).unsqueeze(0).to(device)
            enc   = hf_model.audio_encoder.encode(wav_t, bandwidth=6.0)
            codes = enc.audio_codes[0]  # (1, 9, T_codec)
            codes = codes.squeeze(0).cpu().numpy().T  # (T_codec, 9)

        T_codec = codes.shape[0]
        num_cb  = codes.shape[1]  # 9

        # 4. Apply delay pattern
        # decoder_input_ids[t, k] = codes[t-k, k]  (BOS for t < k)
        T_out   = min(T_codec, max_audio_frames)
        bos     = 1025  # cfg.decoder.bos_token_id
        decoder_input = np.full((T_out, num_cb), bos, dtype=np.int32)
        labels_arr    = np.full((T_out, num_cb), -100, dtype=np.int32)

        for k in range(num_cb):
            for t in range(T_out):
                src_t = t - k
                if src_t >= 0 and src_t < T_codec:
                    decoder_input[t, k] = codes[src_t, k]
                # label at position t for codebook k = codes[t, k] (no delay)
                if t < T_codec:
                    labels_arr[t, k] = codes[t, k]

        np.savez(
            out_path / f"{idx:06d}.npz",
            description_ids=description_ids,
            attention_mask=attention_mask,
            prompt_ids=prompt_ids,
            audio_tokens=decoder_input,
            labels=labels_arr,
        )

        if (idx + 1) % 100 == 0:
            print(f"  {idx + 1} / {len(ds)}", flush=True)

    print(f"[preprocess] Done. {len(ds)} files in {out_path}")
