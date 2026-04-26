# HedgeMamba — Unofficial Implementation

> This is a personal re-implementation of [**"Attention to Mamba: A Recipe for Cross-Architecture Distillation"**](https://arxiv.org/abs/2604.14191) (Moudgil et al., Apple + MILA, April 2026). Not the authors' code, not affiliated with them. Results may differ from the paper.

---

## What I was trying to do

The paper proposes a two-stage recipe for converting a trained Transformer into a Mamba SSM without retraining from scratch — using the Transformer as a teacher and distilling its knowledge into the SSM student.

The appeal: SSMs have **fixed O(1) inference state** regardless of sequence length, whereas Transformers need a KV cache that grows with context. For edge deployment that matters a lot.

I wanted to see if I could get the recipe to work on something I could actually run and evaluate locally.

---

## Why Whisper and not a language model

My first attempt used **Pythia-14m** (a 14M parameter GPT-NeoX model from EleutherAI). The distillation pipeline worked but the results were underwhelming — at 14M parameters the teacher itself is not that capable, so there isn't much to distill. The student converged but the gap between a tiny Transformer and a tiny SSM is small enough that it's hard to tell if the recipe is actually doing anything meaningful.

I switched to **Whisper-tiny** (`openai/whisper-tiny`, the multilingual model) for a more interesting target:

- Speech models process mel spectrograms — the encoder output is a **dense, continuous feature space** (not sparse token embeddings). That gives the distillation loss a much stronger gradient signal.
- The teacher's cross-attention and encoder are already frozen in our student. Only the decoder self-attention gets replaced with the SSM — a clean, contained swap.
- WER is a real, meaningful metric. You can hear whether the student is actually transcribing.

---

## What's in this repo

### Two training backends

| Backend | Entry point | Device |
|---------|-------------|--------|
| PyTorch | `scripts/run_whisper.py` | MPS / CUDA / CPU |
| MLX | `scripts/run_mlx.py` | Apple Silicon only |

The MLX backend is a full re-implementation of the training pipeline — not just inference. It uses MLX's lazy evaluation to fuse the SSM scan loop, giving a meaningful speedup over PyTorch MPS. See `src/mlx/` .

### Distillation in two stages

**Stage 1 — Cosine distillation:** The SSM layers are trained to reproduce the same per-layer decoder hidden states as the Whisper teacher. No transcript supervision yet. This warm-initialises the SSM so Stage 2 doesn't start from a random state.

**Stage 2 — ASR fine-tuning:** Standard cross-entropy on LibriSpeech transcripts, with scheduled sampling that gradually replaces ground-truth decoder tokens with the student's own predictions (0 → 50% over the first half of training). This closes the teacher-forcing gap.

### Architecture

```
Input audio
    │
    ▼
[Whisper Encoder — frozen Conv + 4 Transformer layers]
    │  audio features (1500 × 384)
    ▼
[Decoder — 4 blocks]
  ├─ Self-attention → replaced with HedgeMambaMixer
  ├─ Cross-attention to encoder ← frozen in Stage 1, trained in Stage 2
  ├─ FFN ← frozen in Stage 1, trained in Stage 2
  └─ Layer norms ← frozen in Stage 1, trained in Stage 2
    │
    ▼
Transcript logits (vocab = 51865)
```

The HedgeMambaMixer itself:
- **Hedgehog projection** on B and C: `φ(x) = softmax([Wx, −Wx])` doubles the state dimension and replaces Q/K projections
- **Selective scan** with input-dependent Δt (ZOH discretization)
- **Normalization duplication**: concatenates `[V, ones]` and runs one scan for numerator + denominator
- **SiLU gate** on output
- **Fix-B SSM state caching** for O(1) per-step autoregressive inference

---

## Quick start

```bash
pip install torch transformers datasets jiwer sounddevice
pip install mlx mlx-whisper          # Apple Silicon only
```

### PyTorch training

```bash
# Full run (~6–8 h on M-series)
python scripts/run_whisper.py --device mps

# Quick debug (~5 min, skips Stage 1)
python scripts/run_whisper.py --device mps --debug --skip_stage1
```

### MLX training

```bash
# Full run (~3.5 h on M-series)
python scripts/run_mlx.py

# Seed Stage 2 from an existing PyTorch Stage 1 checkpoint
python scripts/run_mlx.py --skip_stage1 --pt_ckpt checkpoints/whisper_mamba/stage1_final.pt
```

### Inference and benchmarks

```bash
python scripts/mlx_inference.py      # MLX student vs PyTorch teacher latency + WER
python scripts/decoder_bench.py      # PyTorch student decoder speed
python scripts/long_audio_bench.py   # Latency vs decode length (shows O(1) SSM advantage)
python scripts/mic_demo.py           # Live microphone demo
```

---

## Deviations from the paper

| Paper | This repo | Reason |
|-------|-----------|--------|
| RoPE on B and C | Omitted | Whisper already has positional embeddings |
| `state_size = hidden_size` (N = D = 384) | `state_size = 64` (Ns = 128 after Hedgehog) | N = D makes scan state (B, 768, 768) — too slow on MPS |
| Parallel associative scan | Python for-loop | No fused Metal/Triton kernel yet |
| Per-head Hedgehog (H heads × D/H dim) | Single virtual head of size N | Avoids the H × D_h = D constraint when N ≠ D_h |

---

## Results

See [RESULTS.md](RESULTS.md).

## Architecture details

See [DESIGN.md](DESIGN.md).

---

## Repo layout

```
src/
  student/
    hedge_mamba.py       # HedgehogProjection + HedgeMambaMixer (PyTorch)
    whisper_mamba.py     # WhisperMambaStudent + Fix-B state caching
    mlx_hedge_mamba.py   # MLX port (inference + training)
    param_init.py        # Appendix B parameter surgery
    builder.py           # Model factory for Pythia experiments
  mlx/
    model.py             # WhisperMambaMLX — full model for MLX training
    trainer.py           # Stage1Trainer + Stage2Trainer (MLX)
    loss.py              # cosine_distill_loss, ce_loss
    data.py              # LibriSpeech loader → mx.array
    checkpoint.py        # save/load + PT→MLX weight bridge
    utils.py             # LR schedule, grad clipping, WER
  distill/
    whisper_stage1.py    # PyTorch Stage 1 training loop
    whisper_train.py     # PyTorch Stage 2 training loop
    stage1.py            # Generic Stage 1 (Pythia experiments)
    stage2.py            # Generic Stage 2 (Pythia experiments)
  data/
    librispeech.py       # HF datasets + WhisperProcessor loader
  teachers/
    base.py              # Teacher loading + freezing
    gpt_neox.py          # Pythia-specific teacher

scripts/
  run_whisper.py         # PyTorch two-stage training
  run_mlx.py             # MLX two-stage training
  mlx_inference.py       # MLX inference benchmark
  eval_ood.py            # OOD evaluation (LibriSpeech test + FLEURS)
  decoder_bench.py       # Decoder speed + WER
  long_audio_bench.py    # Latency vs decode length
  mic_demo.py            # Live microphone

results/
  ood_wer.json           # Output of eval_ood.py

configs/
  whisper_tiny_to_mamba.yaml
  pythia_70m_to_mamba.yaml
```
