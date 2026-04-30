# Design Notes

Architecture and implementation decisions for this re-implementation of HedgeMamba (arXiv:2604.14191).

Animated walkthroughs of the two training stages:

| Stage 1 — Cosine Distillation | Stage 2 — Scheduled Sampling |
|-------------------------------|-------------------------------|
| ![Stage 1](assets/stage1_cosine_distill.gif) | ![Stage 2](assets/stage2_scheduled_sampling.gif) |

---

## 1. Why two stages

The naive approach — randomly initialise a Mamba student and train it with KL divergence against the teacher — fails badly (validation perplexity > 100 in the paper's ablations). The two-stage recipe exists to avoid this:

**Stage 1** trains the SSM layers to *mimic the teacher's hidden states*, not to predict tokens. This forces the SSM to learn the same representational structure as the attention layers it's replacing — without the instability of autoregressive loss on random weights.

**Stage 2** then fine-tunes the full student on task loss (cross-entropy). The SSM layers start from a warm state rather than random noise, so convergence is fast and stable.

The Hedgehog feature map (Stage 1 linear attention) provides a principled bridge: it shares the same mathematical structure as softmax attention but is a special case of the SSM, so the parameters learned in Stage 1 can be surgically mapped into the SSM's B and C matrices (Appendix B of the paper).

---

## 2. Why I switched from Pythia to Whisper

My first implementation targeted Pythia-14m, a 14M parameter GPT-NeoX language model. The pipeline worked correctly but the results were not interesting:

- A 14M model is weak enough that teacher and random-SSM-student produce similar perplexities anyway
- The distillation signal (difference between teacher and student hidden states) is small
- OpenWebText token representations are sparse — many tokens appear rarely and the cosine loss gradient is noisy

Switching to Whisper-tiny resolved all of these:

- The encoder produces **dense mel-spectrogram features** — every dimension carries information for every input, so cosine similarity between teacher and student hidden states has a strong, consistent gradient
- The task (speech recognition) has a clear binary metric (WER) — you can hear whether the student is working
- The architecture is convenient: encoder is frozen, only the decoder self-attention gets replaced. Clean experiment.

The Pythia code remains in `src/distill/stage1.py`, `stage2.py`, `src/teachers/`, and `configs/pythia_70m_to_mamba.yaml` for reference.

---

## 3. HedgeMamba layer

### 3.1 Hedgehog projection

```
φ(x) = softmax([Wx, −Wx], dim=-1)    x ∈ ℝᴺ → φ(x) ∈ ℝ²ᴺ
```

Applied to the B (keys) and C (queries) projections. This doubles the effective state dimension from N to 2N = Ns.

The doubling is important: the normalization duplication trick (`concat([V, ones])`) uses the extra Ns to carry the denominator of the normalized linear attention in the same scan as the numerator. One scan, two outputs.

### 3.2 Selective SSM

```
h[t] = exp(Δ[t] · A) · h[t-1] + (Δ[t] · u[t]) ⊗ B[t]
y[t] = C[t] · h[t]
```

- `A`: diagonal matrix, stored as `A_log = log(-A)` for numerical stability
- `Δ[t]`: input-dependent, computed as `softplus(dt_proj(dt_rank_proj(x[t])))`
- `B[t]`, `C[t]`: Hedgehog-projected versions of the raw B/C projections
- `u[t]`: value projection `v_proj(x_conv[t])` (after causal depthwise conv)

The discretization is ZOH: `Ā = exp(Δ · A)`, `B̄ = Δ · B`.

### 3.3 Normalization duplication

Instead of:
```
y = (C · h_num) / (C · h_den)
```
which requires two scans, the paper's trick:
```
V_dup = concat([V, ones])     → shape (B, L, 2D)
A_dup = concat([A, A])        → shape (2D, Ns)
```
Run one scan over `V_dup`. Split output into numerator `(B, L, D)` and denominator `(B, L, D)`. Divide: `y = gate * y_num / (|y_den| + ε)`.

### 3.4 Causal depthwise conv

A width-4 causal depthwise convolution applied before the SSM. It mixes local context before the long-range SSM state update — same role as in original Mamba.

PyTorch stores weights as `(D, 1, k)`. MLX needs `(D, k, 1)` — this transposition happens in the PT→MLX weight bridge (`src/mlx/checkpoint.py`).

### 3.5 Fix-B state caching (inference)

During autoregressive generation, the HF generate loop passes one new token per step. A naive SSM would reset `h` to zero each step. Fix-B maintains `h_cache` and `conv_cache` across steps:

```
step 1:  h = None  →  SSM processes prompt, returns h_1
step 2:  h = h_1   →  SSM processes one token, returns h_2
...
```

In PyTorch this is done via `_h_cache` / `_conv_cache` on `WhisperHedgeMambaLayer`. In MLX the mlx-whisper kv_cache mechanism handles it: the SSM returns `(token_counter, (h, conv_ctx))` as its "kv_cache", which the decoder block passes back unchanged on the next step.

---

## 4. Two-stage training

### Stage 1 — cosine distillation

Only the SSM layers train. Everything else (encoder, cross-attention, FFN, layer norms, embeddings) is frozen.

Loss:
```
L₁ = (1/n_layers) Σᵢ mean(1 - cosine_sim(student_hᵢ, teacher_hᵢ, dim=-1))
```

where `hᵢ` is the output of decoder block i (post-residual, post-cross-attention, post-FFN) — not just the self-attention output. Matching full block outputs forces the SSM to produce representations compatible with the frozen cross-attention that sits above it.

Note: MPS and MLX both have float16 precision issues with cosine similarity. Activations must be in float32 before computing the loss.

### Stage 2 — ASR fine-tuning with scheduled sampling

Trained parameters: SSM, cross-attention, FFN, layer norms. Encoder and embeddings stay frozen.

Loss: cross-entropy on next transcript token.

**Scheduled sampling** closes the teacher-forcing gap. During training the decoder sees ground-truth tokens as input; during inference it sees its own predictions. The gap causes a distribution shift.

Fix: replace a fraction `p` of ground-truth decoder tokens with the student's own predictions from a no-grad forward pass:

```
p(step) = ss_max_p × min(1, step / (total_steps × 0.5))
```

`ss_max_p = 0.5` — by the end of training, half the decoder input tokens are self-generated. This forces the student to be robust to its own prediction errors.

---

## 5. MLX port

The full training pipeline has an MLX backend in `src/mlx/`. Key differences from PyTorch:

**No backward() / autograd tape.** MLX uses `nn.value_and_grad(model, loss_fn)`:
```python
loss_and_grad_fn = nn.value_and_grad(student, loss_fn)
loss, grads = loss_and_grad_fn(student, mel, decoder_ids, ...)
optimizer.update(student, grads)
mx.eval(student.parameters(), optimizer.state, loss)
```
`mx.eval()` is critical — it flushes MLX's lazy computation graph and actually executes the Metal kernels.

**Freezing via module methods.** MLX has no `requires_grad=False`. Freezing is done by calling `module.freeze()` / `module.unfreeze()`:
- Stage 1: `model.freeze()` then `block.attn.unfreeze()` per block
- Stage 2: `model.freeze()` then `block.unfreeze()` per block + `decoder.ln.unfreeze()`

**No built-in ignore_index in cross_entropy.** Implemented manually:
```python
valid = (labels_flat >= 0).astype(mx.float32)
safe_labels = mx.clip(labels_flat, 0, V-1)
loss = (cross_entropy(logits, safe_labels, reduction='none') * valid).sum() / valid.sum()
```

**LR scheduling is manual.** MLX optimizers have a settable `learning_rate` attribute; cosine decay with warmup is implemented in `src/mlx/utils.py`.

**Data workers must be 0.** MLX arrays cannot cross process boundaries, so all DataLoaders use `num_workers=0`. Preprocessing runs in the main process but is cached by HuggingFace datasets after the first run.

**Mel format transposition.** PyTorch Whisper uses `(B, 80, T)`, MLX whisper uses `(B, T, 80)`. Transposed in `pt_batch_to_mlx()`.

**SSM scan speedup.** The Python for-loop over sequence positions is fused by MLX's lazy evaluator into fewer Metal kernel dispatches. Measured ~85 ms/step (batch=4, L=448) end-to-end — roughly 8× faster than the PyTorch MPS Python-loop path.

---

## 5.1 Custom Metal scan kernels — experiment and result

### Motivation

In the PyTorch MPS training path the selective SSM scan is an O(L) Python for-loop. PyTorch's eager execution dispatches a separate Metal shader per operation per timestep — ~4 Metal commands × L=448 timesteps = ~1 800 GPU round-trips per scan call, per decoder layer, per training step. This is the dominant bottleneck.

### What we built

Four custom Metal compute kernels written in Metal Shading Language via `mx.fast.metal_kernel` (MLX), wired into `src/ops/selective_scan.py`:

| Kernel | Role |
|--------|------|
| `_fwd_inf_kernel` | Inference forward — no h_states saved |
| `_fwd_train_kernel` | Training forward — saves h_states `(B, L+1, D2, Ns)` for backward |
| `_bwd1_kernel` | Backward recurrence per `(b,d)` threadgroup → `grad_dt`, `grad_u`, `grad_A`, `grad_h0` |
| `_bwd2_kernel` | Parallel `(b,t)` threadgroup → `grad_B`, `grad_C` (no atomics, iterates over D2) |

The backward uses a dual-tree reduction in threadgroup shared memory (`scratch[512]`) to compute `grad_dt` and `grad_u` simultaneously. `grad_A` is accumulated per-batch in `_bwd1_kernel` and summed over the batch dimension in Python.

Autograd integration differs by backend:
- **PyTorch path**: `torch.autograd.Function` (`_SelectiveScanMetal`) with PT↔MLX tensor bridge (`MPS → CPU → MLX → CPU → MPS`)
- **MLX path**: `mx.custom_vjp` with direct `mx.array` I/O — zero copies

### Results

**PyTorch MPS path** (measured on M5 Pro, batch=2, L=448, D2=768, Ns=128):

| Scan | ms / step (fwd+bwd) | Speedup |
|------|---------------------|---------|
| Python for-loop (PyTorch eager) | 688 ms | 1× |
| Metal kernel (via PT↔MLX bridge) | 45 ms | **15.2×** |

**MLX path** (measured on M5 Pro, batch=4, L=448):

| Scan | ms / step (fwd+bwd) | Speedup |
|------|---------------------|---------|
| Python for-loop (MLX lazy eval) | 85 ms | 1× |
| Metal kernel (`mx.custom_vjp`) | 89 ms | **0.96×** (no gain) |

### Why the MLX kernel gives no speedup

The 15.2× gain in the PyTorch path came entirely from eliminating eager Metal dispatch — PyTorch MPS dispatches ~1 800 separate shader invocations per scan, which bottlenecks on CPU-GPU round-trip overhead, not compute.

MLX's lazy evaluation already solves this. When MLX traces the Python for-loop, it builds a computation graph rather than dispatching anything immediately. `mx.eval()` then compiles and launches the whole scan as a small set of fused Metal operations in a single submission — the same thing our explicit kernel does. So the explicit kernel and the Python loop end up dispatching the same Metal workload.

The actual speedup for Apple Silicon training therefore comes from **using MLX at all**, not from writing explicit kernels. The explicit kernels are still used in the PyTorch path (where they matter) and are the fallback for L < 64 bypass logic.

### Running the benchmark

```bash
python scripts/mlx_train_bench.py     # MLX path: Metal kernel vs lazy eval loop
python scripts/train_step_bench.py    # PyTorch MPS path: Metal kernel vs Python loop
```

---

## 6. Parameter surgery (Appendix B)

When transitioning from Stage 1 (Hedgehog linear attention) to Stage 2 (full SSM), the paper maps Hedgehog weights into SSM B/C:

| SSM param | Source | Rationale |
|-----------|--------|-----------|
| `B_proj.weight` | `k_proj.weight` (Hedgehog key) | B encodes input → state; K encodes input → key |
| `C_proj.weight` | `q_proj.weight` (Hedgehog query) | C reads state → output; Q reads state → query |
| `A_log` | initialized to log(1..Ns) | Stable exponential decay at init |
| `conv.weight` | identity kernel | No local mixing at init |
| `gate_proj` | zeros | No gating at init |

This warm start avoids the loss spike that would occur from random SSM initialization after Stage 1.

In this repo the surgery is in `src/student/param_init.py`. It runs automatically when `WhisperMambaStudent` is constructed (the `HedgeMambaMixer` constructor calls `param_init`).

---

## 7. Held-out evaluation — teacher vs student on unseen data

The student was trained on LibriSpeech `train-clean-100` and validated on `validation.clean`. The `test` splits of both the `clean` and `other` configs were never seen during training or used for any hyperparameter decisions. We evaluate both teacher and student on these splits to check whether the distillation generalised beyond the training distribution.

All predictions are normalised to lowercase with punctuation stripped before WER is computed, so formatting differences between teacher (mixed-case) and student (trained on lowercase) don't artificially inflate the teacher's WER.

### Results (300 samples per split, greedy decoding, `repetition_penalty=1.1`)

| Split | Domain | Teacher WER | Student WER | Gap |
|-------|--------|-------------|-------------|-----|
| `test.clean` | Clean audiobooks, diverse speakers | 9.65% | **8.49%** | −1.15% |
| `test.other` | Noisier recordings, more accent variation | 20.23% | **18.0%** | −2.23% |

### What the numbers say

**The student outperforms the teacher on both splits.** This is the headline result and it holds up on the harder `test.other` split where the gap is larger (2.2 pp vs 1.2 pp). A few things explain it:

1. **Fine-tuning on LibriSpeech.** The original Whisper-tiny was trained on a broad mix of web audio. The student was explicitly fine-tuned on LibriSpeech transcriptions — lowercase, clean, audiobook-style. The test splits come from the same distribution, so the student is better matched to the reference format even after normalisation.

2. **Scheduled sampling closes the exposure gap.** The teacher is evaluated autoregressively but was never trained with its own predictions as decoder input. The student's Stage 2 scheduled sampling explicitly prepares it for this — by the end of training half the decoder input tokens are the student's own predictions, so it's robust to its own errors in a way the teacher is not.

3. **Task specialisation.** The teacher is `openai/whisper-tiny` — the multilingual model (99 languages, 51,865-token vocab). The student inherits the same architecture but was fine-tuned on English LibriSpeech transcriptions only. Fine-tuning a multilingual model on a single-language task typically improves WER on that task, and that effect carries through to the student.

The WER gap grows from clean (−1.2%) to other (−2.2%), which suggests the student's robustness advantage is larger under harder conditions. This is somewhat surprising — one might expect the SSM's fixed-state approximation to degrade more on harder speech — but it does not, at least within LibriSpeech-style audiobooks.

### Sample transcription comparison

A few representative examples from `test.clean`:

| Reference | Teacher | Student |
|-----------|---------|---------|
| concord returned to its place | **conquered** returned to its place | **coloncore** returned to its place |
| congratulations were poured in upon the princess | congratulations **report** in upon the princess | **conggradulations** were poured in upon the princess |
| you will be frank with me i always am | ✓ correct | ✓ correct |
| can you imagine why buckingham has been so violent | can you imagine **my** buckingham | can you imagine **my** buckingham |

Both models struggle with the same rare proper nouns and uncommon words. The error patterns are similar in character — substitutions, not insertions — which is consistent with two models that have learned the same underlying language representation but with slightly different fine-tuning.

### What this evaluation does not cover

The test splits are still clean read speech (audiobooks). They are genuinely held-out but not out-of-domain in the full sense. The student has not been evaluated on:

- **Spontaneous speech** — conversational, disfluent, with false starts and filler words
- **Telephone / noisy conditions** — CHiME, NOISY student challenge  
- **Accented English** — non-native speakers, regional dialects
- **Different domains** — meetings (AMI), broadcast news, medical dictation

Performance on those would likely be worse for both teacher and student, and the relative gap could go either way. That evaluation is left for future work (`scripts/eval_ood.py` is ready; it just needs a dataset that loads cleanly from HuggingFace without auth).

---

## 8. Known gaps and future work

**Python scan loop (resolved).** We wrote explicit Metal kernels (`src/ops/selective_scan.py`) and integrated them via `mx.custom_vjp` in the MLX path and `torch.autograd.Function` in the PyTorch path. The kernels give 15.2× speedup in the PyTorch MPS path. In the MLX path they provide no additional speedup because MLX's lazy evaluator already fuses the Python loop into the same Metal workload — see Section 5.1 for the full writeup.

**Greedy decoding only.** The MLX inference uses mlx-whisper's `dec.decode` which doesn't support beam search. The PyTorch student is evaluated with `repetition_penalty=1.1` but no beam search either. Adding beam search to the MLX path would likely close some of the WER gap vs the paper.

**Evaluation scope is narrow.** See Section 7 for held-out test results on LibriSpeech `test` splits. The teacher (`openai/whisper-tiny`) is multilingual (99 languages); the student was fine-tuned on English only. Cross-lingual capability after English fine-tuning has not been properly measured — a rigorous evaluation would need a clean Hindi (or other language) benchmark with proper normalisation and a baseline that rules out WER inflation from script mismatch. Other untested domains: spontaneous speech, telephone/noisy conditions, accented English, meetings.

**Pythia distillation incomplete.** The LM distillation pipeline (Stage 1 + Stage 2 on OpenWebText) was validated for correctness but not run to completion. The teacher (Pythia-14m) is too small to produce meaningful numbers. Pythia-70m or larger would be the right next experiment.
