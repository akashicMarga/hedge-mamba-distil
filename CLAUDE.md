# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial implementation of HedgeMamba (arXiv:2604.14191) — cross-architecture distillation that converts Transformer attention layers to SSM (State Space Model) mixers. Primary use case: distilling Whisper-tiny (speech recognition) into a faster student with HedgeMamba SSM decoders.

## Commands

```bash
# Install dependencies
pip install torch transformers datasets jiwer sounddevice
pip install mlx mlx-whisper  # Apple Silicon only

# Quick debug run (~5 min, skips Stage 1)
python scripts/run_whisper.py --device mps --debug --skip_stage1

# Full 2-stage Whisper distillation (~6-8 h on M-series)
python scripts/run_whisper.py --device mps

# Resume from Stage 2 checkpoint
python scripts/run_whisper.py --device mps --resume_stage2

# MLX inference benchmark (Apple Silicon, ~3.7× faster than PyTorch)
python scripts/mlx_inference.py

# Decoder latency benchmark
python scripts/decoder_bench.py
python scripts/long_audio_bench.py  # latency vs decode length

# Live mic demo
python scripts/mic_demo.py

# Pythia/LM distillation
python scripts/run_stage1.py
python scripts/run_stage2.py
```

No test suite exists. Validate correctness by running `--debug` mode (token budget: 10M).

## Architecture

### Two-Stage Distillation Pipeline

The Whisper pipeline (`scripts/run_whisper.py`) orchestrates two stages:

1. **Stage 1** (`src/distill/whisper_stage1.py`): Cosine distillation loss between student and teacher decoder layer activations. Only SSM weights are trained; everything else frozen. Warm-initializes SSM layers without transcript supervision.

2. **Stage 2** (`src/distill/whisper_train.py`): Cross-entropy loss on transcript tokens, with **scheduled sampling** (0%→50% ground-truth token replacement during training) to close the teacher-forcing/autoregressive gap. Trains SSM, cross-attn, FFN, and layer norms.

### Student Model Components

**`src/student/hedge_mamba.py`** — Core SSM. Key pieces:
- `HedgehogProjection`: φ(x) = softmax([Wx, −Wx], dim=-1) — kernel feature map replacing Q/K. This doubles effective state size.
- `HedgeMambaMixer`: Input-dependent Δt, SiLU gate, selective SSM scan
- `HedgeMambaLayer`: Drop-in attention replacement

**`src/student/whisper_mamba.py`** — Whisper adapter:
- `WhisperHedgeMambaLayer`: Adapts Whisper's attention interface to HedgeMamba
- `WhisperMambaStudent`: Frozen encoder + SSM decoder. Has `h_cache` / `conv_cache` for O(1) per-step autoregressive inference (Fix B from paper Appendix)

**`src/student/param_init.py`** — Parameter surgery (Appendix B): maps Hedgehog weights from teacher attention projections (B_proj ← k_proj, C_proj ← q_proj) so SSM layers start from a warm init instead of random.

**`src/student/builder.py`** — `HedgeMambaStudent`: deep-copies teacher, then replaces attention layers with HedgeMamba.

**`src/student/mlx_hedge_mamba.py`** — MLX backend (inference only). Lazy evaluation fuses the Python scan loop, yielding ~2.3× fewer Metal kernel dispatches.

### Data & Evaluation

- `src/data/librispeech.py`: `make_librispeech_loaders` returns `(train_loader, val_loader, processor)` using HuggingFace datasets + WhisperProcessor for mel spectrogram extraction
- `src/eval/`: perplexity, latency benchmarks, lm-evaluation-harness integration
- `src/teachers/`: teacher model loading and freezing utilities

### Config Files

- `configs/whisper_tiny_to_mamba.yaml`: `state_size: 64` (→128 after Hedgehog doubling), 2 Stage 1 epochs + 5 Stage 2 epochs, `ss_max_p: 0.5`
- `configs/pythia_70m_to_mamba.yaml`: Pythia-14m teacher, LM distillation

## Key Design Decisions & Gotchas

- **float16 on MPS**: Activations must be cast to float32 before computing cosine loss; MPS has silent precision issues with float16 reductions. The Hedgehog doubling makes effective state size 2×`state_size`.

- **State size**: Paper uses N=D (state = hidden), but N=64 is used here to fit MPS memory. `state_size` in config refers to pre-Hedgehog size.

- **RoPE omitted**: Whisper already uses positional embeddings; adding RoPE would double-count.

- **Python scan loop**: The selective SSM scan is an O(L) Python loop — correct but slow during training. A fused Metal/Triton kernel would improve throughput significantly.


- **Token budgets**: Stage 1 validates at 100M tokens, debug at 10M. Stage 2 trains to 500M tokens (full) or 10M (debug).

- **Checkpoints**: `checkpoints/whisper_mamba/stage1_final.pt`, `stage2_final.pt`, `whisper_mamba_final.pt`. TensorBoard logs in `runs/`.
