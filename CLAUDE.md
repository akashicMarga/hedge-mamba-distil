# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Unofficial implementation of HedgeMamba (arXiv:2604.14191) — cross-architecture distillation that converts Transformer attention layers to SSM (State Space Model) mixers. Two distillation targets: Whisper-tiny (speech recognition) and IndicParlerTTS (text-to-speech, 9-codebook DAC decoder).

## Commands

```bash
# Install dependencies
pip install torch transformers datasets jiwer sounddevice
pip install mlx mlx-whisper  # Apple Silicon only

# ── Whisper distillation ──────────────────────────────────────────────────────

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

# ── ParlerTTS distillation (requires mlx-audio-train) ────────────────────────
# Dataset: ai4b-hf/GLOBE-annotated — exact training data for indic-parler-tts
#   567K train / 14K test samples, 90 GB audio, 18 languages including all
#   major Indic languages. Requires HF login: huggingface-cli login

# Step 0: preprocess dataset into .npz cache (run once per split)
# Full multilingual (90 GB, ~12 h on M-series)
python scripts/run_parler_preprocess.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --split train --out_dir ./data/parler_distil
python scripts/run_parler_preprocess.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --split test --out_dir ./data/parler_distil

# Indic-only subset (e.g. Hindi only, much smaller)
python scripts/run_parler_preprocess.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --split train --lang_filter Hindi --out_dir ./data/parler_distil_hi
python scripts/run_parler_preprocess.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --split test --lang_filter Hindi --out_dir ./data/parler_distil_hi

# Quick smoke-test (100 samples, ~2 min)
python scripts/run_parler_preprocess.py --mlx_audio_train /path/to/mlx-audio-train \
    --debug --split train --out_dir ./data/parler_distil_debug

# Stage 1: cosine distillation, SSM warm-start (~3 epochs)
python scripts/run_parler_stage1.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --train_cache ./data/parler_distil/train \
    --val_cache ./data/parler_distil/validation \
    --config configs/parler_tts_to_mamba.yaml

# Stage 1 debug (1 epoch, quick)
python scripts/run_parler_stage1.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --train_cache ./data/parler_distil_debug/train \
    --val_cache ./data/parler_distil_debug/validation \
    --debug

# Stage 2: 9-codebook CE + scheduled sampling (~5 epochs)
python scripts/run_parler_stage2.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --train_cache ./data/parler_distil/train \
    --val_cache ./data/parler_distil/validation \
    --stage1_ckpt ./checkpoints/parler_mamba/stage1_epoch_3 \
    --config configs/parler_tts_to_mamba.yaml

# Audio quality eval (MCD + RTF vs teacher)
python scripts/eval_parler.py \
    --mlx_audio_train /path/to/mlx-audio-train \
    --stage2_ckpt ./checkpoints/parler_mamba/stage2_epoch_5

# AR step-latency benchmark (O(1) SSM vs O(L) attention)
python scripts/benchmark_parler_ar.py
```

No test suite exists. Validate correctness by running `--debug` mode (Whisper: 10M token budget; Parler: 1 epoch on 100 samples).

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

### ParlerTTS Pipeline (MLX-only, Apple Silicon)

Teacher: `IndicParlerTTS` from `mlx-audio-train`. All training uses the MLX path — no HF `parler_tts` package needed.

**`src/student/parler_mamba.py`** — PyTorch reference model (unused during training, kept for export/debugging).

**`src/mlx/parler_model.py`** — `ParlerMambaMLX`: 24-layer decoder with HedgeMamba self-attn. Key functions:
- `apply_weight_surgery_mlx(student, teacher)`: warm-starts x_proj B/C slots from k_proj/q_proj, copies v_proj/out_proj (Appendix B for the MLX architecture)
- `encode_description(ids, mask)`: T5 encoder, called once outside grad-fn and shared between teacher and student
- `build_first_emb(decoder, prompt_emb, audio_tokens)`: builds [prompt | audio] + positional embeddings

**`src/mlx/parler_trainer.py`** — Stage 1 (cosine distillation) + Stage 2 (9-codebook CE + scheduled sampling). Encoder computed once per batch outside grad-fn to halve T5 compute.

**`src/mlx/parler_data.py`** — `preprocess_and_cache()`: DAC-encodes waveforms via `IndicParlerTTS.audio_encoder`, applies delay pattern (decoder_input[t,k] = codec[t-k,k]), caches to .npz. `make_parler_loaders()` reads the cache.

**`scripts/eval_parler.py`** — Mel Cepstral Distortion (MCD, dB) + Real-Time Factor vs teacher. Requires `pip install librosa`.

### Config Files

- `configs/whisper_tiny_to_mamba.yaml`: `state_size: 64` (→128 after Hedgehog doubling), 2 Stage 1 epochs + 5 Stage 2 epochs, `ss_max_p: 0.5`
- `configs/parler_tts_to_mamba.yaml`: `state_size: null` (→2048, N=D paper default), 3 Stage 1 + 5 Stage 2 epochs, `ss_max_p: 0.5`, 9 codebooks, max_seq_len 512
- `configs/pythia_70m_to_mamba.yaml`: Pythia-14m teacher, LM distillation

## Key Design Decisions & Gotchas

- **float16 on MPS**: Activations must be cast to float32 before computing cosine loss; MPS has silent precision issues with float16 reductions. The Hedgehog doubling makes effective state size 2×`state_size`.

- **State size**: Paper uses N=D (state = hidden), but N=64 is used here to fit MPS memory. `state_size` in config refers to pre-Hedgehog size.

- **RoPE omitted**: Whisper already uses positional embeddings; adding RoPE would double-count.

- **Python scan loop**: The selective SSM scan is an O(L) Python loop — correct but slow during training. A fused Metal/Triton kernel would improve throughput significantly.


- **Token budgets**: Stage 1 validates at 100M tokens, debug at 10M. Stage 2 trains to 500M tokens (full) or 10M (debug).

- **Checkpoints**: `checkpoints/whisper_mamba/stage1_final.pt`, `stage2_final.pt`, `whisper_mamba_final.pt`. TensorBoard logs in `runs/`.

- **Parler delay pattern**: The 9-codebook DAC decoder uses an interleaved delay — codebook k's input at step t is the token from step t-k. `preprocess_and_cache()` pre-applies this so the dataloader returns already-shifted `audio_tokens` and `labels`. Do not apply the delay again inside the model.

- **Parler x_proj surgery**: MLX Hedgehog projects from N (post-x_proj) not from D (pre-x_proj) as in PyTorch. Surgery therefore warm-starts `x_proj.weight[dt_rank:]` (the B/C slots) from teacher k_proj/q_proj instead of hhog_k/q.phi directly.

- **Parler audio encoder API**: `preprocess_and_cache()` probes `model.audio_encoder`, `model.dac`, `model.codec` in order. If mlx-audio-train changes this attribute name, update the probe list in `parler_data.preprocess_and_cache`.

- **DAC sample rate**: Descript Audio Codec expects 44100 Hz. `preprocess_and_cache()` resamples automatically via librosa (preferred) or scipy. Dataset audio at other rates (e.g. 16 kHz) is handled correctly.
