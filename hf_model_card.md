---
license: mit
language:
- en
tags:
- automatic-speech-recognition
- whisper
- mamba
- ssm
- distillation
- hedgehog
- cross-architecture-distillation
- librispeech
- apple-silicon
- mlx
base_model: openai/whisper-tiny
datasets:
- librispeech_asr
metrics:
- wer
---

# WhisperMamba — HedgeMamba Distillation of Whisper-tiny

Unofficial implementation of [**"Attention to Mamba: A Recipe for Cross-Architecture Distillation"**](https://arxiv.org/abs/2604.14191) (Moudgil et al., Apple + MILA, April 2026) applied to Whisper-tiny.

The decoder's self-attention layers are replaced with **HedgeMamba SSM mixers** using two-stage knowledge distillation. The encoder and cross-attention remain frozen from the original Whisper-tiny weights.

> Not the authors' code, not affiliated with Apple or MILA.

**Code:** [github.com/akashicMarga/hedge-mamba-distil](https://github.com/akashicMarga/hedge-mamba-distil)

---

## What this model is

Whisper-tiny has 4 decoder layers, each with a self-attention block. This student replaces every self-attention with a **HedgeMambaMixer** — a selective SSM with:

- **Hedgehog projection** on B and C: `φ(x) = softmax([Wx, −Wx])` — doubles effective state size and replaces Q/K
- **Selective scan** with input-dependent Δt (ZOH discretization)
- **SiLU gate** on the output
- **Fix-B state caching** for O(1) per-step autoregressive inference (no KV cache growth)

The encoder (4 Transformer layers + Conv frontend) is fully frozen. Only the decoder SSM weights are learned from scratch.

---

## Files in this repo

| File | Description |
|------|-------------|
| `pytorch/whisper_mamba_final.pt` | Final PyTorch state dict (Stage 1 + Stage 2, 144 MB) |
| `pytorch/stage1_final.pt` | Stage 1 only (cosine-distilled SSM, before ASR fine-tuning) |
| `mlx/whisper_mamba_mlx_final.npz` | Final MLX weights (142 MB, Apple Silicon inference) |
| `mlx/whisper_mamba_mlx_final.json` | MLX checkpoint metadata |

The `.pt` files are raw `state_dict` `OrderedDict`s — load with `torch.load(..., map_location="cpu")`. The `.npz` is an MLX weight archive — load with `mlx.core.load(...)`.

---

## Results

### WER on LibriSpeech test splits (greedy decoding, lowercase, no punctuation)

| Model | Split | WER |
|-------|-------|-----|
| Whisper-tiny teacher | test.clean | 9.65% |
| **WhisperMamba student (PyTorch)** | test.clean | **8.49%** |
| Whisper-tiny teacher | test.other | 20.23% |
| **WhisperMamba student (PyTorch)** | test.other | **18.0%** |

The student outperforms the teacher on both splits. The larger gap on `test.other` suggests scheduled sampling gives the student better robustness to its own decoding errors.

### Validation WER during Stage 2 (PyTorch, LibriSpeech train-clean-100)

| Epoch | Val WER |
|-------|---------|
| 3 | — |
| 5 | ~5% |

### Inference latency (single utterance, 20 samples, Apple M-series)

| Model | Backend | Latency |
|-------|---------|---------|
| Whisper-tiny teacher | PyTorch MPS | ~154 ms |
| WhisperMamba student | PyTorch MPS | ~129 ms |
| **WhisperMamba student** | **MLX** | **~41 ms** |

MLX is ~3.7× faster than the PyTorch teacher. The O(1) SSM state means latency does not grow with sequence length (unlike the KV cache in standard Whisper).

---

## Training

### Two-stage distillation

**Stage 1 — Cosine distillation** (warm-up, ~3 h on M-series):
- Loss: layer-wise cosine similarity between student and teacher decoder hidden states
- Only SSM weights trained; everything else frozen
- Warm-initializes SSM from teacher attention projections (Appendix B parameter surgery: `B_proj ← k_proj`, `C_proj ← q_proj`)
- 2 epochs, LibriSpeech train-clean-100, batch size 8

**Stage 2 — ASR fine-tuning** (~5 h on M-series):
- Loss: cross-entropy on LibriSpeech transcripts
- Scheduled sampling: ground-truth token replacement ramps 0% → 50% over first half of training, closing the teacher-forcing gap
- SSM, cross-attn, FFN, and layer norms all trained
- 5 epochs, LibriSpeech train-clean-100, batch size 8

An MLX re-implementation trains both stages end-to-end in ~3.5 h.

### Config

```yaml
teacher:   openai/whisper-tiny
state_size: 64          # ×2 after Hedgehog = 128 effective
batch_size: 8
stage1_lr:  0.0005
stage2_lr:  0.0001
ss_max_p:   0.5         # scheduled sampling ceiling
stage1_epochs: 2
stage2_epochs: 5
data: librispeech_asr train.100 / validation
```

---

## Usage

Install the source repo, then load the checkpoint:

```bash
pip install torch transformers datasets jiwer
git clone https://github.com/akashicMarga/hedge-mamba-distil
cd hedge-mamba-distil
```

```python
import torch
from src.student.whisper_mamba import WhisperMambaStudent

# Load the state dict
state_dict = torch.load("pytorch/whisper_mamba_final.pt", map_location="cpu")

# Rebuild the student (requires the source repo)
from transformers import WhisperProcessor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")

model = WhisperMambaStudent.from_teacher("openai/whisper-tiny", state_size=64)
model.load_state_dict(state_dict)
model.eval()
```

For the MLX backend (Apple Silicon):

```bash
pip install mlx mlx-whisper
python scripts/mlx_inference.py   # benchmarks student vs teacher
python scripts/mic_demo.py        # live microphone
```

---

## Deviations from the paper

| Paper | This repo | Reason |
|-------|-----------|--------|
| RoPE on B and C | Omitted | Whisper already has positional embeddings |
| `state_size = hidden_size` (N = D = 384) | `state_size = 64` (128 after Hedgehog) | N = D makes scan state (B, 768, 768) — too slow on MPS |
| Parallel associative scan | Python for-loop | No fused Metal/Triton kernel yet |
| Per-head Hedgehog (H heads × D/H dim) | Single virtual head of size N | Avoids the H × D_h = D constraint when N ≠ D_h |

---

## Citation

```bibtex
@misc{moudgil2026hedgemamba,
  title   = {Attention to Mamba: A Recipe for Cross-Architecture Distillation},
  author  = {Moudgil, Abhinav and others},
  year    = {2026},
  url     = {https://arxiv.org/abs/2604.14191}
}
```
