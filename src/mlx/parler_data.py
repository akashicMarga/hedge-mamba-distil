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


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


try:
    import torch
    from torch.utils.data import Dataset, DataLoader
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False
    Dataset = object   # placeholder so the class definition parses


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
):
    """Return (train_loader, val_loader).

    num_workers=0 required — MLX arrays cannot cross process boundaries.
    """
    if not _TORCH_AVAILABLE:
        raise ImportError("torch is required for make_parler_loaders. Install it in your training env.")
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
    hf_dataset_id: str = "ai4b-hf/GLOBE-annotated",
    out_dir: str = "./data/parler_distil",
    split: str = "train",
    max_samples: Optional[int] = None,
    max_audio_len_s: float = 6.0,
    description_col: str = "auto",
    lang_filter: Optional[str] = None,
) -> None:
    """Pre-process a HuggingFace Parler TTS dataset into .npz cache files.

    Requires:
        pip install datasets transformers
        mlx-audio-train on sys.path (for IndicParlerTTS + MLX DAC encoder)

    No HF parler_tts package needed — all encoding uses mlx-audio-train's
    IndicParlerTTS directly.

    Args:
        hf_dataset_id:   HF dataset id. Default is ai4b-hf/GLOBE-annotated
                         (the exact data indic-parler-tts was trained on).
        description_col: Column name for the style description text.
                         "auto" probes common names: text_description,
                         description, caption. GLOBE-annotated uses
                         "text_description".
        lang_filter:     If set, keep only rows where sample["language"]
                         equals this value (e.g. "Hindi", "Tamil").
                         None = keep all languages.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    # Lazy import — mlx-audio-train path must be set by caller
    try:
        import mlx.core as mx
        import models  # noqa: F401 — confirms mlx-audio-train is on sys.path
    except ImportError as e:
        raise ImportError(
            "mlx-audio-train not found on sys.path. "
            "Add it with: sys.path.insert(0, '/path/to/mlx-audio-train')"
        ) from e

    out_path = Path(out_dir) / split
    out_path.mkdir(parents=True, exist_ok=True)

    # Strategy:
    #   max_samples ≤ 1000  → streaming (fast, no download needed)
    #   max_samples > 1000 or None → targeted data_files download of only N shards.
    #     Avoids the full-dataset (90 GB) download while being faster than streaming
    #     for thousands of samples (xet streaming deadlocks on large datasets).
    _SHARD_COUNTS = {"train": 184, "test": 5, "validation": 5}
    total_shards = _SHARD_COUNTS.get(split, 184)

    use_streaming = max_samples is not None and max_samples <= 1000
    if use_streaming:
        print(
            f"[preprocess] Loading dataset {hf_dataset_id} / {split} (streaming) ...",
            flush=True,
        )
        ds = load_dataset(hf_dataset_id, split=split, streaming=True)
        if lang_filter:
            ds = ds.filter(lambda x: x.get("language", "") == lang_filter)
        if max_samples:
            ds = ds.take(max_samples)
    else:
        # Download only enough shards to cover max_samples rows.
        samples_per_shard = {"train": 3084, "test": 2861}.get(split, 3084)
        n_shards = min(
            int((max_samples or 10_000) / samples_per_shard) + 3,
            total_shards,
        )
        print(
            f"[preprocess] Loading dataset {hf_dataset_id} / {split}: "
            f"downloading {n_shards}/{total_shards} shards ...",
            flush=True,
        )
        data_files = {
            split: [
                f"data/{split}-{i:05d}-of-{total_shards:05d}.parquet"
                for i in range(n_shards)
            ]
        }
        ds = load_dataset(
            hf_dataset_id,
            data_files=data_files,
            split=split,
            verification_mode="no_checks",
        )
        if lang_filter:
            ds = ds.filter(lambda x: x.get("language", "") == lang_filter)
            print(f"[preprocess] After lang_filter='{lang_filter}': {len(ds)} samples", flush=True)
        if max_samples and len(ds) > max_samples:
            ds = ds.select(range(max_samples))

    # Resolve description column name
    _DESC_CANDIDATES = ["text_description", "description", "caption", "prompt"]
    if description_col == "auto":
        first = next(iter(ds))
        description_col = next(
            (c for c in _DESC_CANDIDATES if c in first),
            None,
        )
        if description_col is None:
            raise KeyError(
                f"Cannot find a description column. Tried {_DESC_CANDIDATES}. "
                f"Available columns: {list(first.keys())}"
            )
        print(f"[preprocess] Using description column: '{description_col}'", flush=True)

    # DAC encoder — use mlx-audio-train's own implementation which loads
    # weights directly from the HF safetensors (no extra pip package needed).
    try:
        import importlib.util, os as _os
        _script = _os.path.join(
            _os.path.dirname(importlib.util.find_spec("models").origin),
            "..", "scripts", "preprocess_bhojpuri_dac.py"
        )
        import importlib.util as _ilu
        _spec = _ilu.spec_from_file_location("_dac_preprocess", _script)
        _dac_mod = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_dac_mod)
        _load_hf_weights = _dac_mod._load_hf_weights
        _build_dac_encoder = _dac_mod._build_dac_encoder
        _encode_audio = _dac_mod._encode_audio
    except Exception as e:
        raise ImportError(
            f"Could not load DAC encoder from mlx-audio-train/scripts/preprocess_bhojpuri_dac.py: {e}"
        ) from e

    print("[preprocess] Loading DAC encoder weights from ai4bharat/indic-parler-tts ...")
    _weights = _load_hf_weights("ai4bharat/indic-parler-tts")
    _dac_enc, _dac_quant = _build_dac_encoder(_weights)
    del _weights  # free memory
    print("[preprocess] DAC encoder ready")

    # T5 tokenizer — from HF hub, no parler_tts package needed
    t5_tokenizer = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")

    # DAC expected sample rate
    DAC_SR = 44100
    max_audio_frames = int(max_audio_len_s * DAC_SR / 512)

    n_desc = len(ds) if not use_streaming else (max_samples or "?")
    print(f"[preprocess] Processing {n_desc} samples → {out_path}")
    for idx, sample in enumerate(ds):
        # 1. T5-tokenize description
        desc_enc = t5_tokenizer(
            sample[description_col], return_tensors="np",
            truncation=True, max_length=256,
        )
        description_ids = desc_enc["input_ids"][0].astype(np.int32)
        attention_mask  = desc_enc["attention_mask"][0].astype(np.int32)

        # 2. Tokenize prompt text (text to speak)
        pmt_enc = t5_tokenizer(
            sample["text"], return_tensors="np",
            truncation=True, max_length=128,
        )
        prompt_ids = pmt_enc["input_ids"][0].astype(np.int32)

        # 3. Resample to 44100 Hz if needed, then DAC-encode
        audio_arr = np.array(sample["audio"]["array"], dtype=np.float32)
        sr        = sample["audio"]["sampling_rate"]
        if sr != DAC_SR:
            from math import gcd
            from scipy.signal import resample_poly
            g         = gcd(int(sr), DAC_SR)
            audio_arr = resample_poly(audio_arr, DAC_SR // g, sr // g).astype(np.float32)

        # _encode_audio returns (9, T_frames) int16
        codes_9T  = _encode_audio(_dac_enc, _dac_quant, audio_arr)  # (9, T)
        codes     = codes_9T.T.astype(np.int32)                     # (T, 9)

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
            print(f"  {idx + 1} done", flush=True)

    print(f"[preprocess] Done. {idx + 1} files in {out_path}")
