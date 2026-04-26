# Results

WER = word error rate (lower is better). Greedy decoding, `repetition_penalty=1.1`. All text normalised to lowercase with punctuation stripped before scoring.

---

## WER — Whisper-tiny teacher vs WhisperMamba student

| Model | Backend | WER |
|-------|---------|-----|
| Whisper-tiny (teacher) | PyTorch MPS | ~25%† |
| WhisperMamba student | PyTorch MPS | **~5%** |
| WhisperMamba student | MLX (from scratch) | 10.3% |
| WhisperMamba student | MLX (PT Stage 1 seed) | 9.9% |

† The teacher WER appears high because LibriSpeech references are lowercase/unpunctuated and the teacher outputs mixed-case with punctuation. The student is fine-tuned on lowercased text so its output format matches the references directly.

---

## Inference latency — PyTorch MPS vs MLX

Measured on a single utterance, averaged over 20 samples after 3 warmup runs.

| Model | Backend | Latency |
|-------|---------|---------|
| Whisper-tiny (teacher) | PyTorch MPS | ~154 ms |
| WhisperMamba student | PyTorch MPS | ~129 ms |
| **WhisperMamba student** | **MLX** | **~41 ms** |

MLX speedup over PyTorch teacher: **~3.7×**  
MLX speedup over PyTorch student: **~3.1×**

---

## Stage 1 training — cosine distillation loss

PyTorch run (2 epochs, 7,136 steps, LibriSpeech train-clean-100, batch size 8):

| Step | Val cos-loss |
|------|-------------|
| 500 | 0.0302 |
| 1500 | 0.0177 |
| 3000 | 0.0132 |
| 5000 | 0.0107 |
| 7000 | 0.0098 |
| final | 0.0117 |

MLX run (same config):

| Step | Val cos-loss |
|------|-------------|
| 500 | 0.0302 |
| 3500 | 0.0122 |
| 7000 | 0.0098 |
| final | 0.0117 |

---

## Stage 2 training — WER by epoch

PyTorch run (5 epochs):

| Epoch | WER |
|-------|-----|
| 1 | — |
| 3 | — |
| 5 | **~5%** |

MLX run from scratch (5 epochs):

| Epoch | WER |
|-------|-----|
| 1 | 13.6% |
| 2 | 10.6% |
| 3 | 10.0% |
| 4 | 10.0% |
| 5 | 10.3% |

MLX run seeded from PT Stage 1 checkpoint (5 epochs):

| Epoch | WER |
|-------|-----|
| 1 | 11.2% |
| 2 | 10.3% |
| 3 | 10.1% |
| 4 | 10.1% |
| 5 | **9.9%** |

---

## Held-out evaluation — unseen test splits

Evaluated on LibriSpeech `test` splits that were never used during training or validation (student trained on `train-clean-100` only). 300 samples each, PyTorch MPS, greedy decoding.

| Split | Difficulty | Teacher WER | Student WER | Gap |
|-------|-----------|-------------|-------------|-----|
| `test.clean` | Clean audiobooks | 9.65% | **8.49%** | −1.15% |
| `test.other` | Noisier, more accent variation | 20.23% | **18.0%** | −2.23% |

The student outperforms the teacher on both splits. The gap is larger on the harder split, which suggests the Stage 2 scheduled sampling gives the student better robustness to its own decoding errors than the teacher has out-of-the-box. See DESIGN.md §7 for a detailed breakdown and sample transcriptions.

---

## Pythia experiments (early, incomplete)

Tested on Pythia-14m → HedgeMamba-14m distillation on OpenWebText.

These runs validated the pipeline but did not produce strong results — the teacher model is too small for the distillation signal to be meaningful. Perplexity numbers not reported; see the Pythia configs and `src/distill/stage1.py` / `stage2.py` if you want to reproduce.

---

## Training hardware and time

| Run | Hardware | Time |
|-----|----------|------|
| PyTorch Stage 1 (2 epochs) | Apple M-series, MPS | ~3 h |
| PyTorch Stage 2 (5 epochs) | Apple M-series, MPS | ~5 h |
| MLX Stage 1 + Stage 2 (full) | Apple M-series | ~3.5 h |
| MLX Stage 2 only (PT seed) | Apple M-series | ~2.5 h |
