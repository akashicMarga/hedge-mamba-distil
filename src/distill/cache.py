"""Pre-cache teacher layer outputs to disk.

Strategy: single teacher forward pass per batch (no grad), outputs stored as float16.
Stage 1 training reads targets from cache — teacher never runs during training loop.
2–3× throughput improvement; removes need to hold both models in memory simultaneously.

Speed knobs:
  - cache_batch_size: use a larger batch than training (e.g. 32 vs 8) — biggest win
  - num_workers: parallel data loading
  - compile_teacher: torch.compile the teacher for ~30% faster inference

File layout per shard:
  {cache_dir}/shard_{i:05d}.pt  →  dict{ "input_ids": Tensor, "layer_outputs": list[Tensor] }
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm


def cache_teacher_outputs(
    teacher,
    dataloader: DataLoader,
    cache_dir: str | Path,
    max_shards: int | None = None,
    dtype: torch.dtype = torch.float16,
    compile_teacher: bool = True,
) -> None:
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Check for existing shards so we can resume interrupted caching runs
    existing = sorted(cache_dir.glob("shard_*.pt"))
    start_idx = len(existing)
    if start_idx > 0:
        print(f"Resuming cache from shard {start_idx} ({start_idx} shards already done)")

    teacher.eval()

    # torch.compile: helps on CUDA, hurts on MPS (60s warmup with no benefit)
    if compile_teacher:
        device_str = str(next(teacher._model.parameters()).device)
        if "mps" in device_str:
            print("torch.compile skipped (MPS — use CUDA for compile speedup)")
        else:
            try:
                teacher._model = torch.compile(teacher._model, mode="reduce-overhead")
                print("Teacher compiled with torch.compile ✓", flush=True)
            except Exception as e:
                print(f"torch.compile skipped ({e})")

    teacher.register_layer_hooks()
    shard_idx = 0

    try:
        for batch in tqdm(dataloader, desc="Caching teacher outputs"):
            if max_shards is not None and shard_idx >= max_shards:
                break
            # Skip shards already written (resume support)
            if shard_idx < start_idx:
                shard_idx += 1
                continue

            input_ids = batch["input_ids"]
            _ = teacher.forward(input_ids)
            layer_outputs = [t.to(dtype).cpu() for t in teacher.get_layer_outputs()]

            shard_path = cache_dir / f"shard_{shard_idx:05d}.pt"
            torch.save({"input_ids": input_ids.cpu(), "layer_outputs": layer_outputs}, shard_path)
            shard_idx += 1
    finally:
        teacher.remove_hooks()

    print(f"Cached {shard_idx} shards to {cache_dir}", flush=True)


class CachedTeacherDataset(torch.utils.data.Dataset):
    def __init__(self, cache_dir: str | Path):
        self.cache_dir = Path(cache_dir)
        self.shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No cached shards found in {cache_dir}")

    def __len__(self) -> int:
        return len(self.shards)

    def __getitem__(self, idx: int) -> dict:
        return torch.load(self.shards[idx], weights_only=True)
