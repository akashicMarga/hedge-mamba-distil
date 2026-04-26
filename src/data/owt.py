"""OpenWebText dataloaders.

Two modes:
  1. Streaming (OpenWebTextDataset) — tokenises on-the-fly from HuggingFace.
     Slow: tokeniser is the bottleneck. Good for debug runs.

  2. Fast (TokenBinDataset) — reads a pre-tokenised binary file written by
     scripts/pretokenize.py. Zero Python overhead; GPU-bound, not CPU-bound.
     Use this for any run > 10M tokens.
"""
import torch
import numpy as np
from torch.utils.data import IterableDataset, Dataset, DataLoader
from transformers import AutoTokenizer
from pathlib import Path


class OpenWebTextDataset(IterableDataset):
    def __init__(
        self,
        tokenizer_id: str = "EleutherAI/pythia-70m",
        seq_len: int = 1024,
        split: str = "train",
        cache_dir: str | None = None,
        max_tokens: int | None = None,
    ):
        from datasets import load_dataset
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        self.seq_len = seq_len
        self.max_tokens = max_tokens
        self.dataset = load_dataset(
            "Skylion007/openwebtext", split=split, streaming=True, cache_dir=cache_dir
        )

    def __iter__(self):
        # Worker sharding: with num_workers > 0, each DataLoader worker gets a
        # different slice of the stream so data is not duplicated.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Skip every (num_workers - worker_id) examples so workers interleave
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        buf = []
        tokens_yielded = 0
        for doc_idx, example in enumerate(self.dataset):
            # Each worker handles its own stripe of documents
            if doc_idx % num_workers != worker_id:
                continue
            ids = self.tokenizer.encode(example["text"], add_special_tokens=False)
            buf.extend(ids)
            while len(buf) >= self.seq_len + 1:
                chunk = buf[: self.seq_len + 1]
                buf = buf[self.seq_len + 1:]
                yield {"input_ids": torch.tensor(chunk[: self.seq_len], dtype=torch.long)}
                tokens_yielded += self.seq_len
                if self.max_tokens is not None and tokens_yielded >= self.max_tokens:
                    return


def make_dataloader(
    tokenizer_id: str = "EleutherAI/pythia-70m",
    seq_len: int = 1024,
    batch_size: int = 8,
    split: str = "train",
    cache_dir: str | None = None,
    max_tokens: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    ds = OpenWebTextDataset(
        tokenizer_id=tokenizer_id,
        seq_len=seq_len,
        split=split,
        cache_dir=cache_dir,
        max_tokens=max_tokens,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=num_workers)


class TokenBinDataset(Dataset):
    """Fast map-style dataset over a pre-tokenised .bin file (uint16 mmap).
    Run scripts/pretokenize.py once to create the file.
    Supports multi-worker DataLoader safely (each worker reads its own slice).

    Use offset_tokens / max_tokens to carve out train vs val slices:
      train = TokenBinDataset(bin, max_tokens=580_000_000)
      val   = TokenBinDataset(bin, offset_tokens=580_000_000)  # last 20M
    """

    def __init__(
        self,
        bin_path: str | Path,
        seq_len: int = 1024,
        offset_tokens: int = 0,
        max_tokens: int | None = None,
    ):
        self.seq_len = seq_len
        arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
        arr = arr[offset_tokens:]
        if max_tokens is not None:
            arr = arr[:max_tokens]
        # Number of full non-overlapping sequences
        n_seqs = len(arr) // (seq_len + 1)
        self.data = arr[:n_seqs * (seq_len + 1)].reshape(n_seqs, seq_len + 1)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        row = self.data[idx]
        return {"input_ids": torch.from_numpy(row[:self.seq_len].astype(np.int64))}


def make_fast_dataloader(
    bin_path: str | Path,
    seq_len: int = 1024,
    batch_size: int = 32,
    offset_tokens: int = 0,
    max_tokens: int | None = None,
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: bool = True,
) -> DataLoader:
    """Fast dataloader from pre-tokenised binary. Use for any run > 10M tokens."""
    ds = TokenBinDataset(bin_path, seq_len=seq_len, offset_tokens=offset_tokens, max_tokens=max_tokens)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
        drop_last=True,
    )
