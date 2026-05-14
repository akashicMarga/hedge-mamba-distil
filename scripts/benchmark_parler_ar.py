"""AR step-latency benchmark: ParlerDecoder (attention) vs ParlerMambaMLX (SSM).

Measures single decoder-step wall-clock time after warming up a KV cache
(teacher) or SSM state (student) to a given history length.  Reports
median latency and the O(1) vs O(L) speedup ratio.

Key insight (from DESIGN.md §5.1):
  Attention KV cache: each AR step reads L cached keys/values → O(L).
  SSM recurrent state: each AR step updates a fixed (B, 2D, Ns) tensor
  → O(1) regardless of sequence length.

  At 10 s audio / 44100 Hz / 512 frames = ~860 steps:
  The attention component shrinks from O(860) to O(1) per step per layer.

Usage
-----
    # From hedge-mamba-distil root:
    export PYTHONPATH=/path/to/mlx-audio-train:$PYTHONPATH
    python scripts/benchmark_parler_ar.py
    python scripts/benchmark_parler_ar.py --steps 100 200 400 860 --reps 20
"""
import argparse
import sys
import time
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# Lazy imports (mlx-audio-train on sys.path)
# ---------------------------------------------------------------------------


def _load_models(state_size: int):
    """Load teacher (ParlerDecoder) and build student (ParlerMambaMLX).

    Uses random weights — no HF download required for benchmarking.
    """
    from models.indic_parler_tts.config import DecoderConfig, IndicParlerTTSConfig
    from models.indic_parler_tts.decoder import ParlerDecoder
    from models.indic_parler_tts.model import IndicParlerTTS
    from src.mlx.parler_model import ParlerMambaMLX

    cfg = IndicParlerTTSConfig()
    # Build lightweight teacher decoder only (no T5/DAC for benchmark)
    teacher_dec = ParlerDecoder(cfg.decoder)
    mx.eval(teacher_dec.parameters())

    # Build student: needs a full IndicParlerTTS shell for ParlerMambaMLX
    # Use a minimal shell with the decoder only
    teacher_full = IndicParlerTTS(cfg)
    mx.eval(teacher_full.parameters())

    import copy
    student_base = IndicParlerTTS(cfg)
    student_base.load_weights(
        list(dict(mx.utils.tree_flatten(teacher_full.parameters())).items()),
        strict=False,
    )
    student = ParlerMambaMLX(student_base, state_size=state_size)
    mx.eval(student.parameters())

    return teacher_dec, student, cfg.decoder


# ---------------------------------------------------------------------------
# Warm-up helpers
# ---------------------------------------------------------------------------


def _make_enc_hidden(cfg, T_enc: int = 32):
    return mx.random.normal((1, T_enc, cfg.hidden_size))


def _run_teacher_step(teacher_dec, enc_hidden, bos, step, self_caches, cross_caches, cfg):
    emb = teacher_dec.embed_audio(bos, offset=step)
    out = teacher_dec.forward_layers(
        emb, enc_hidden, mask=None,
        self_caches=self_caches, cross_caches=cross_caches,
    )
    return out


def _run_student_step(student, enc_hidden, bos_emb, self_caches, cross_caches):
    # Student uses the model's decoder directly for the step
    from src.mlx.parler_model import _causal_mask
    out = student.model.decoder.forward_layers(
        bos_emb, enc_hidden, mask=None,
        self_caches=self_caches, cross_caches=cross_caches,
    )
    return out


def _warm_up(decoder, enc_hidden, n_steps, cfg, is_mamba: bool):
    """Run n_steps to populate caches, return (self_caches, cross_caches)."""
    from models.indic_parler_tts.model import _causal_mask

    self_caches  = [[] for _ in range(cfg.num_layers)]
    cross_caches = [[] for _ in range(cfg.num_layers)]
    bos = mx.full((1, cfg.num_codebooks), cfg.bos_token_id, dtype=mx.int32)

    for step in range(n_steps):
        emb = decoder.embed_audio(bos, offset=step)
        if step == 0:
            T   = emb.shape[1]
            mask = _causal_mask(T, dtype=emb.dtype)
        else:
            mask = None
        out = decoder.forward_layers(
            emb, enc_hidden, mask=mask,
            self_caches=self_caches, cross_caches=cross_caches,
        )
    mx.eval(out)
    return self_caches, cross_caches


def _measure_step(decoder, enc_hidden, caches, step, cfg, n_reps: int) -> float:
    self_caches, cross_caches = caches
    bos = mx.full((1, cfg.num_codebooks), cfg.bos_token_id, dtype=mx.int32)
    times = []
    for _ in range(n_reps):
        emb = decoder.embed_audio(bos, offset=step)
        t0  = time.perf_counter()
        out = decoder.forward_layers(
            emb, enc_hidden, mask=None,
            self_caches=self_caches, cross_caches=cross_caches,
        )
        mx.eval(out)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.median(times))


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------


def run_benchmark(
    step_lengths: list[int],
    warm_reps: int,
    measure_reps: int,
    state_size: int,
):
    print("Loading models (random weights, no HF download) ...", flush=True)
    teacher_dec, student, cfg = _load_models(state_size)
    enc_hidden = _make_enc_hidden(cfg)

    print(f"\nConfig: D={cfg.hidden_size}, H={cfg.num_heads}, L={cfg.num_layers}")
    print(f"SSM state_size={state_size}, Ns={2*state_size} (after Hedgehog)")
    print()
    print(f"{'History':>8}  {'Attn ms':>10}  {'SSM ms':>9}  "
          f"{'Speedup':>9}  {'KV MB':>8}  {'SSM MB':>8}")
    print("-" * 67)

    results = []
    for n in step_lengths:
        # ── Teacher (attention) ───────────────────────────────────────
        t_caches = _warm_up(teacher_dec, enc_hidden, n, cfg, is_mamba=False)
        # Warm-up measurement kernel
        for _ in range(warm_reps):
            _measure_step(teacher_dec, enc_hidden, t_caches, n, cfg, n_reps=1)
        attn_ms = _measure_step(
            teacher_dec, enc_hidden, t_caches, n, cfg, n_reps=measure_reps
        )

        # KV cache memory: 2 tensors * num_layers, each (1, H, n, head_dim)
        head_dim = cfg.hidden_size // cfg.num_heads
        kv_mb = (
            2 * cfg.num_layers * 1 * cfg.num_heads * n * head_dim * 2  # float16 bytes
        ) / 1e6

        # ── Student (SSM) ────────────────────────────────────────────
        mamba_dec = student.model.decoder
        m_caches  = _warm_up(mamba_dec, enc_hidden, n, cfg, is_mamba=True)
        for _ in range(warm_reps):
            _measure_step(mamba_dec, enc_hidden, m_caches, n, cfg, n_reps=1)
        mamba_ms = _measure_step(
            mamba_dec, enc_hidden, m_caches, n, cfg, n_reps=measure_reps
        )

        # SSM state memory: fixed per layer
        D  = cfg.hidden_size
        Ns = 2 * state_size
        ssm_mb = (cfg.num_layers * (2 * D * Ns) * 2) / 1e6  # float32 or float16

        speedup = attn_ms / mamba_ms if mamba_ms > 0 else float("nan")
        print(f"{n:>8}  {attn_ms:>10.2f}  {mamba_ms:>9.2f}  "
              f"{speedup:>9.2f}×  {kv_mb:>8.1f}  {ssm_mb:>8.1f}")
        results.append(dict(n=n, attn_ms=attn_ms, mamba_ms=mamba_ms,
                            speedup=speedup, kv_mb=kv_mb, ssm_mb=ssm_mb))

    print()
    print("Attention KV cache grows O(n_steps); SSM state is fixed.")
    peak = results[-1]
    print(f"At {peak['n']} steps: {peak['attn_ms']:.1f} ms → {peak['mamba_ms']:.1f} ms "
          f"({peak['speedup']:.1f}× speedup), KV cache {peak['kv_mb']:.0f} MB "
          f"vs SSM {peak['ssm_mb']:.0f} MB (fixed).")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parler AR benchmark: attention KV-cache vs SSM O(1)"
    )
    parser.add_argument(
        "--mlx_audio_train", default=None,
        help="Path to mlx-audio-train repo root (adds to sys.path)",
    )
    parser.add_argument(
        "--steps", type=int, nargs="+",
        default=[50, 100, 200, 400, 600, 860],
        help="History lengths to benchmark (AR steps already generated)",
    )
    parser.add_argument("--warm",       type=int, default=3,
                        help="Warm-up reps per measurement point")
    parser.add_argument("--reps",       type=int, default=15,
                        help="Measurement reps per point")
    parser.add_argument("--state_size", type=int, default=64,
                        help="SSM state_size (64=fast, 256=quality, 1024=paper)")
    args = parser.parse_args()

    if args.mlx_audio_train:
        sys.path.insert(0, args.mlx_audio_train)
    else:
        # Try common relative location
        candidate = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "../mlx-audio-train"
        )
        if os.path.isdir(candidate):
            sys.path.insert(0, os.path.abspath(candidate))

    run_benchmark(args.steps, args.warm, args.reps, args.state_size)
