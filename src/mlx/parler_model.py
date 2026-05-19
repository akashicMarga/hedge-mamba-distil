"""ParlerMambaMLX — full model for MLX distillation of IndicParlerTTS.

Mirrors src/mlx/model.py (WhisperMambaMLX) but for the 24-layer Parler
audio decoder:
  encoder (T5):  frozen — provides cross-attention conditioning
  decoder:       24 transformer layers, each self-attn replaced by
                 HedgeMambaMixerMLX (arXiv:2604.14191)

Dependency on mlx-audio-train
------------------------------
ParlerMambaMLX wraps an IndicParlerTTS instance loaded from
mlx-audio-train.  Add mlx-audio-train to sys.path before importing:

    import sys; sys.path.insert(0, "/path/to/mlx-audio-train")
    from models.indic_parler_tts.model import IndicParlerTTS
    teacher = IndicParlerTTS.from_pretrained()
    student = ParlerMambaMLX(teacher_copy, state_size=64)

The teacher and student are separate IndicParlerTTS instances so hooks
and weight surgery never corrupt the reference model.

Forward signature
-----------------
Unlike WhisperMambaMLX which runs the encoder internally, ParlerMambaMLX
splits encode + decode because the T5 encoder is large (24 layers) and
produces the same enc_hidden for both teacher and student in Stage 1.
Pre-computing enc_hidden once per batch halves the T5 compute:

    enc_hidden = student.encode_description(description_ids)
    prompt_emb = student.encode_prompt(prompt_ids)
    first_emb  = build_first_emb(student, prompt_emb, audio_tokens)
    logits, hiddens = student(enc_hidden, first_emb)
"""
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from src.student.mlx_hedge_mamba import HedgeMambaMixerMLX


# ---------------------------------------------------------------------------
# Drop-in self-attention replacement
# ---------------------------------------------------------------------------


class ParlerMambaSelfAttnMLX(nn.Module):
    """HedgeMambaMixerMLX wrapped for ParlerSelfAttention's interface.

    ParlerSelfAttention is called as:
        out = layer.self_attn(layer.self_attn_ln(x), mask=mask, cache=self_cache)
    where cache is a plain Python list:
        []             — first call, cache gets populated
        [k_t, v_t]     — subsequent calls (attention KV cache)

    We keep the same list-based cache protocol but store the SSM state:
        []                    — first call
        [mlx_kv_cache]        — (token_ctr, (h, conv)) from HedgeMambaMixerMLX
    """

    def __init__(self, dim: int, num_heads: int, state_size: int = 64):
        super().__init__()
        self.mixer = HedgeMambaMixerMLX(
            dim=dim, num_heads=num_heads, state_size=state_size
        )

    def __call__(
        self,
        x: mx.array,
        mask: mx.array = None,
        cache: list = None,
    ) -> mx.array:
        """
        x    : (B, L, D)
        cache: [] or [ssm_kv_cache]   (list, mutated in-place)
        Returns (B, L, D) — same as ParlerSelfAttention.
        """
        kv = cache[0] if (cache is not None and len(cache) > 0) else None
        out, new_kv, _ = self.mixer(x, mask=mask, kv_cache=kv)
        if cache is not None:
            if len(cache) == 0:
                cache.append(new_kv)
            else:
                cache[0] = new_kv
        return out


# ---------------------------------------------------------------------------
# Full student model
# ---------------------------------------------------------------------------


class ParlerMambaMLX(nn.Module):
    """IndicParlerTTS backbone with HedgeMamba SSM decoder.

    Usage::

        teacher = IndicParlerTTS.from_pretrained()   # mlx-audio-train
        student = ParlerMambaMLX(teacher_copy, state_size=64)
        student.freeze_for_stage1()
        print(student.param_summary())
    """

    def __init__(self, parler_model, state_size: int = 64):
        """
        parler_model : IndicParlerTTS instance (from mlx-audio-train).
                       Passed by reference — caller should pass a fresh
                       load, not the teacher instance.
        state_size   : SSM state per channel before Hedgehog doubling.
                       64 → Ns=128 (fast); 256 → Ns=512 (quality);
                       None / 1024 → Ns=2048 (paper default, N=D).
        """
        super().__init__()
        self.model = parler_model   # IndicParlerTTS
        self.state_size = state_size or parler_model.cfg.decoder.hidden_size

        cfg = parler_model.cfg.decoder
        D   = cfg.hidden_size   # 1024
        H   = cfg.num_heads     # 16

        # Replace every decoder self-attention with HedgeMamba SSM
        for layer in self.model.decoder.layers:
            layer.self_attn = ParlerMambaSelfAttnMLX(D, H, state_size=self.state_size)

        # Remove DACDecoder from MLX module tree — it stores weights in raw
        # Python lists (not nn.Module children), which breaks trainable_parameters()
        # and update(). DAC is not needed during training (only for inference).
        # Stored separately so eval scripts can still access it.
        self._dac = self.model.pop('dac', None)

        # Default: Stage 2 freeze profile (safe default for evaluation)
        self.freeze_for_stage2()

    # ── Freeze helpers (α la WhisperMambaMLX) ─────────────────────────────────

    def freeze_for_stage1(self) -> None:
        """Stage 1: only SSM layers (layer.self_attn) are trainable."""
        # Must freeze from student root so student.trainable_parameters() is correct
        self.freeze()
        for layer in self.model.decoder.layers:
            layer.self_attn.unfreeze()

    def freeze_for_stage2(self) -> None:
        """Stage 2: decoder layers + final_ln train; T5 + embeddings frozen."""
        self.freeze()
        for layer in self.model.decoder.layers:
            layer.unfreeze()
        self.model.decoder.final_ln.unfreeze()
        self.model.decoder.lm_heads.unfreeze()
        # Keep frozen: codebook embeddings, position embeddings
        for emb in self.model.decoder.embed_tokens:
            emb.freeze()
        self.model.decoder.embed_positions.freeze()

    # ── Encode helpers (called outside the grad-fn for efficiency) ──────────

    def encode_description(
        self,
        description_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        """T5 encoder (frozen). Returns enc_hidden (B, T_desc, 1024)."""
        if attention_mask is not None:
            try:
                return self.model.text_encoder(description_ids, attention_mask=attention_mask)
            except TypeError:
                pass
        return self.model.text_encoder(description_ids)

    def encode_prompt(self, prompt_ids: mx.array) -> mx.array:
        """embed_prompts (frozen). Returns prompt_emb (B, T_prompt, 1024)."""
        return self.model.embed_prompts(prompt_ids)

    # ── Forward (inside the grad-fn) ────────────────────────────────────────

    def __call__(
        self,
        enc_hidden: mx.array,   # (B, T_desc, D)  — pre-computed, frozen
        first_emb:  mx.array,   # (B, T_seq,  D)  — prompt + audio embeddings
    ) -> tuple[mx.array, list[mx.array]]:
        """
        enc_hidden : T5 output, pre-computed outside grad-fn.
        first_emb  : decoder input embeddings (prompt_emb + audio_emb) with
                     position embeddings already added.

        Returns:
            logits       : (B, T_audio, num_codebooks * lm_vocab_size)
            hidden_states: list[(B, T_seq, D)], one per decoder block
                           — used for Stage 1 cosine distillation loss
        """
        T   = first_emb.shape[1]
        mask = _causal_mask(T, dtype=first_emb.dtype)

        x = first_emb
        hidden_states: list[mx.array] = []

        for layer in self.model.decoder.layers:
            # No caches: training runs full sequence in one forward pass
            x = layer(x, enc_hidden, mask=mask,
                      self_cache=None, cross_cache=None)
            hidden_states.append(x)

        x      = self.model.decoder.final_ln(x)
        logits = self.model.decoder.lm_heads(x)   # (B, T, 9*1088)
        return logits, hidden_states

    # ── Utilities ────────────────────────────────────────────────────────────

    def param_summary(self) -> str:
        total = sum(v.size for _, v in tree_flatten(self.model.parameters())) / 1e6
        trainable = sum(
            v.size for _, v in tree_flatten(self.model.trainable_parameters())
        ) / 1e6
        return (
            f"Total: {total:.1f}M  |  "
            f"Trainable: {trainable:.1f}M  |  "
            f"Frozen: {total - trainable:.1f}M"
        )


# ---------------------------------------------------------------------------
# Teacher forward  (α la get_teacher_hiddens in src/mlx/model.py)
# ---------------------------------------------------------------------------


def get_parler_teacher_hiddens(
    teacher,          # IndicParlerTTS with original ParlerDecoder (attention)
    enc_hidden: mx.array,  # (B, T_desc, D)
    first_emb:  mx.array,  # (B, T_seq,  D)
) -> list[mx.array]:
    """Run the teacher decoder (attention), collecting post-block hidden states.

    Called outside nn.value_and_grad (no gradients needed for teacher).
    Returns list of (B, T_seq, D), one per decoder block — same indexing
    as student hidden_states so cosine_distill_loss works directly.
    """
    T    = first_emb.shape[1]
    mask = _causal_mask(T, dtype=first_emb.dtype)

    x = first_emb
    hidden_states: list[mx.array] = []

    for layer in teacher.decoder.layers:
        # No caches: teacher runs full sequence (no SSM state to maintain)
        x = layer(x, enc_hidden, mask=mask,
                  self_cache=None, cross_cache=None)
        hidden_states.append(x)

    return hidden_states


# ---------------------------------------------------------------------------
# Embedding helpers
# ---------------------------------------------------------------------------


def build_first_emb(
    decoder,          # ParlerDecoder (student.model.decoder)
    prompt_emb: mx.array,   # (B, T_prompt, D)  from embed_prompts
    audio_tokens: mx.array, # (B, T_audio,  num_codebooks) int32
    prompt_offset: int = 0,
) -> mx.array:
    """Build the full decoder input embedding for training.

    Mirrors the prefill construction in IndicParlerTTS.generate():
        first_emb = [prompt_emb | audio_emb] + position_embeddings

    audio_tokens : delayed audio token matrix from the dataset,
                   shape (B, T_audio, 9), dtype int32.

    Returns (B, T_prompt + T_audio, D).
    """
    B, T_audio, num_cb = audio_tokens.shape
    # Sum codebook embeddings for each audio frame: (B, T_audio, D)
    audio_emb = None
    for k in range(num_cb):
        e = decoder.embed_tokens[k](audio_tokens[:, :, k])  # (B, T_audio, D)
        audio_emb = e if audio_emb is None else audio_emb + e

    full_emb = mx.concatenate([prompt_emb, audio_emb], axis=1)  # (B, T_total, D)
    T_total  = full_emb.shape[1]
    pos      = decoder.embed_positions.weight[prompt_offset : prompt_offset + T_total]
    return full_emb + pos[None]  # broadcast over batch


# ---------------------------------------------------------------------------
# Weight surgery  (paper Appendix B, MLX edition)
# ---------------------------------------------------------------------------


def apply_weight_surgery_mlx(student: ParlerMambaMLX, teacher) -> None:
    """Warm-start SSM weights from teacher self-attention projections.

    MLX architecture differs from PyTorch: hhog_k/q operate on the N-dim
    x_proj output (not on D-dim hidden directly).  We therefore warm-start
    x_proj's B/C slots instead of hhog.phi, and copy v/out projections.

    Mapping (per layer):
      x_proj.weight[dt_rank : dt_rank+N, :] <- teacher k_proj.weight[:N, :]
      x_proj.weight[dt_rank+N :,          ] <- teacher q_proj.weight[:N, :]
      v_proj.weight                          <- teacher v_proj.weight
      out_proj.weight                        <- teacher out_proj.weight
      conv_weight[:, -1, :]                  <- 1.0  (near-identity causal conv)

    Teacher self-attn attribute names are probed at runtime because
    mlx-audio-train may use q_proj/k_proj or q/k conventions.
    """
    def _get_attn(t_layer, name: str):
        attn = t_layer.self_attn
        for attr in (name, name.replace("_proj", ""), name[0]):
            if hasattr(attn, attr):
                return getattr(attn, attr)
        raise AttributeError(
            f"Teacher self_attn has no attribute matching '{name}'. "
            f"Available: {[a for a in dir(attn) if not a.startswith('_')]}"
        )

    t_layers = teacher.decoder.layers
    s_layers = student.model.decoder.layers

    for t_layer, s_layer in zip(t_layers, s_layers):
        mixer   = s_layer.self_attn.mixer   # HedgeMambaMixerMLX
        N       = mixer.state_size
        dt_rank = mixer.dt_rank

        try:
            t_k = _get_attn(t_layer, "k_proj")
            t_q = _get_attn(t_layer, "q_proj")
            t_v = _get_attn(t_layer, "v_proj")
            t_o = _get_attn(t_layer, "out_proj")
        except AttributeError as e:
            print(f"[surgery] WARNING: {e} — skipping weight surgery for this layer")
            return

        # x_proj warm-start: B slot ← k_proj[:N], C slot ← q_proj[:N]
        xw = mixer.x_proj.weight                       # (dt_rank+2N, D) in MLX
        xw_np = xw.astype(mx.float32)
        k_rows = t_k.weight.astype(mx.float32)[:N, :]  # (N, D)
        q_rows = t_q.weight.astype(mx.float32)[:N, :]  # (N, D)

        new_xw = mx.concatenate([
            xw_np[:dt_rank, :],                        # dt rows unchanged
            k_rows,                                    # B slot
            q_rows,                                    # C slot
        ], axis=0)
        mixer.x_proj.weight = new_xw

        # v_proj and out_proj copy directly (D×D → D×D)
        mixer.v_proj.weight   = t_v.weight.astype(mx.float32)
        mixer.out_proj.weight = t_o.weight.astype(mx.float32)

        # Near-identity causal conv: last kernel position = 1, rest = 0
        D, k, _ = mixer.conv_weight.shape
        conv_new = mx.zeros((D, k, 1))
        # Set last kernel slice to identity
        eye_col  = mx.ones((D, 1, 1))
        conv_new = mx.concatenate(
            [mx.zeros((D, k - 1, 1)), eye_col], axis=1
        )
        mixer.conv_weight = conv_new

    mx.eval(student.parameters())
    print("[surgery] Weight surgery complete — SSM warm-started from teacher attention", flush=True)


# ---------------------------------------------------------------------------
# Causal mask (same as model.py in mlx-audio-train)
# ---------------------------------------------------------------------------


def _causal_mask(T: int, dtype=mx.float32) -> mx.array:
    return mx.triu(mx.full((T, T), -1e9, dtype=dtype), k=1)[None, None, :, :]
