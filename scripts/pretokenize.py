#!/usr/bin/env python
"""Pre-tokenise OpenWebText once to disk as packed uint16 tensors.

Run this once before any training. Output is a memory-mapped binary file
that loads as raw token IDs with zero Python overhead.

Usage:
  python scripts/pretokenize.py --tokens 200_000_000 --out data_cache/owt_tokens.bin
"""
import argparse
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=200_000_000,
                        help="Total tokens to write (default 200M — covers S1+S2 validate run)")
    parser.add_argument("--out", default="data_cache/owt_tokens.bin")
    parser.add_argument("--tokenizer", default="EleutherAI/pythia-70m")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = load_dataset("Skylion007/openwebtext", split="train", streaming=True)

    print(f"Tokenising up to {args.tokens:,} tokens → {out}")

    # Write to a pre-allocated numpy mmap
    arr = np.memmap(out, dtype=np.uint16, mode="w+", shape=(args.tokens,))

    written = 0
    buf = []
    pbar = tqdm(total=args.tokens, unit="tok", unit_scale=True)

    for example in ds:
        ids = tokenizer.encode(example["text"], add_special_tokens=False)
        buf.extend(ids)
        # Flush in large chunks
        if len(buf) >= 1_000_000:
            chunk = buf[:1_000_000]
            buf = buf[1_000_000:]
            end = min(written + len(chunk), args.tokens)
            arr[written:end] = chunk[:end - written]
            pbar.update(end - written)
            written = end
            if written >= args.tokens:
                break

    # Flush remainder
    if written < args.tokens and buf:
        end = min(written + len(buf), args.tokens)
        arr[written:end] = buf[:end - written]
        pbar.update(end - written)
        written = end

    arr.flush()
    pbar.close()
    print(f"Done. Written {written:,} tokens to {out}  ({out.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
