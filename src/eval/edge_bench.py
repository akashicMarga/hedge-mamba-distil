"""Edge benchmark: tok/s, peak memory, latency vs context length."""
import time
import torch
from dataclasses import dataclass


@dataclass
class BenchResult:
    context_len: int
    tokens_per_sec: float
    peak_memory_mb: float
    latency_ms: float


@torch.no_grad()
def bench_model(
    model,
    tokenizer,
    context_lengths: list[int] = [128, 512, 1024, 2048, 4096],
    n_tokens_gen: int = 64,
    device: str = "cpu",
) -> list[BenchResult]:
    model.eval()
    model.to(device)
    results = []

    for ctx_len in context_lengths:
        input_ids = torch.randint(0, tokenizer.vocab_size, (1, ctx_len), device=device)

        if device == "mps":
            torch.mps.empty_cache()
            mem_before = 0  # MPS doesn't expose peak alloc easily
        elif device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats(device)
            mem_before = torch.cuda.memory_allocated(device)

        t0 = time.perf_counter()
        for _ in range(n_tokens_gen):
            out = model(input_ids)
            logits = out.logits if hasattr(out, "logits") else out
            next_tok = logits[:, -1].argmax(-1, keepdim=True)
            input_ids = torch.cat([input_ids[:, 1:], next_tok], dim=1)
        elapsed = time.perf_counter() - t0

        if device.startswith("cuda"):
            peak_mb = (torch.cuda.max_memory_allocated(device) - mem_before) / 1e6
        else:
            peak_mb = 0.0

        results.append(BenchResult(
            context_len=ctx_len,
            tokens_per_sec=n_tokens_gen / elapsed,
            peak_memory_mb=peak_mb,
            latency_ms=elapsed * 1000 / n_tokens_gen,
        ))
        print(f"ctx={ctx_len:5d} | {n_tokens_gen / elapsed:.1f} tok/s | {elapsed*1000/n_tokens_gen:.1f} ms/tok")

    return results
