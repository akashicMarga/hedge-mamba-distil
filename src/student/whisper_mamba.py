"""WhisperMamba: Whisper encoder (frozen) + HedgeMamba SSM decoder.

Architecture:
  Encoder : frozen Whisper-tiny (Conv + 4 transformer layers).
  Decoder : each self-attention layer replaced with HedgeMambaMixer
            (paper-faithful, arXiv:2604.14191).
            Cross-attention, FFN, layer norms, embeddings unchanged.

Fix B — SSM state caching during generation:
  HF 5.x passes the decoder ONE new token per step when use_cache=True.
  WhisperHedgeMambaLayer stores (h_cache, conv_cache) across steps so
  the SSM never resets to zero mid-utterance.
  Cache is reset between utterances via generate().
"""
import copy
import torch
import torch.nn as nn
from .hedge_mamba import HedgeMambaMixer


class WhisperHedgeMambaLayer(nn.Module):
    """Drop-in replacement for Whisper decoder self-attention.

    Wraps HedgeMambaMixer and handles:
      • Whisper attention interface  (hidden_states, **kwargs) → (out, None)
      • Fix B inference caching      (_h_cache, _conv_cache)
    """

    def __init__(self, dim: int, num_heads: int, state_size: int | None = None):
        super().__init__()
        self.mixer = HedgeMambaMixer(dim=dim, num_heads=num_heads,
                                     state_size=state_size)
        # Fix B caches — shapes set on first forward; None between utterances
        self._h_cache:    torch.Tensor | None = None   # (B, 2D, Ns)
        self._conv_cache: torch.Tensor | None = None   # (B, k-1, D)
        self._caching_active: bool = False

    def reset_ssm_cache(self) -> None:
        self._h_cache    = None
        self._conv_cache = None

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # All Whisper attention kwargs (past_key_values, attention_mask,
        # cache_position, output_attentions, …) are accepted and ignored.
        h_prev    = self._h_cache    if self._caching_active else None
        conv_prev = self._conv_cache if self._caching_active else None

        out, h, new_conv = self.mixer(hidden_states,
                                      h_prev=h_prev,
                                      conv_cache=conv_prev)

        if self._caching_active:
            self._h_cache    = h.detach()
            self._conv_cache = new_conv.detach()

        return out, None


class WhisperMambaStudent(nn.Module):
    """Full WhisperMamba model: frozen Whisper encoder + HedgeMamba decoder.

    Usage:
        whisper = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
        student = WhisperMambaStudent(whisper, state_size=None)   # None = dim (paper default)
        student = student.to(device)

    Forward:
        out = student(input_features=..., decoder_input_ids=...)
        logits = out.logits   # (B, L, vocab_size)

    Generation (Fix B — O(L) per step):
        ids = student.generate(input_features=..., language="en", task="transcribe")
        text = processor.batch_decode(ids, skip_special_tokens=True)
    """

    def __init__(self, whisper_model, state_size: int | None = None):
        super().__init__()
        self.backbone = copy.deepcopy(whisper_model)
        self.config   = whisper_model.config

        # Replace every decoder self-attention with HedgeMambaMixer
        for layer in self.backbone.model.decoder.layers:
            dim       = layer.self_attn.q_proj.in_features
            num_heads = layer.self_attn.num_heads
            layer.self_attn = WhisperHedgeMambaLayer(
                dim=dim, num_heads=num_heads, state_size=state_size
            )

        # Freeze encoder (identical to teacher — no benefit retraining it)
        for p in self.backbone.model.encoder.parameters():
            p.requires_grad = False

        # Freeze token + position embeddings (shared vocab with teacher)
        for p in self.backbone.model.decoder.embed_tokens.parameters():
            p.requires_grad = False
        for p in self.backbone.model.decoder.embed_positions.parameters():
            p.requires_grad = False

    def forward(self, input_features, decoder_input_ids, labels=None):
        return self.backbone(
            input_features=input_features,
            decoder_input_ids=decoder_input_ids,
            labels=labels,
        )

    # ── Fix B helpers ─────────────────────────────────────────────────────

    def _set_ssm_caching(self, active: bool) -> None:
        for layer in self.backbone.model.decoder.layers:
            if isinstance(layer.self_attn, WhisperHedgeMambaLayer):
                layer.self_attn._caching_active = active
                layer.self_attn.reset_ssm_cache()

    def generate(self, input_features, **kwargs):
        """Generate with Fix B SSM state caching (O(L) per decode step)."""
        self._set_ssm_caching(True)
        try:
            result = self.backbone.generate(
                input_features=input_features, **kwargs
            )
        finally:
            self._set_ssm_caching(False)
        return result

    # ── Utilities ─────────────────────────────────────────────────────────

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad) / 1e6
        return (
            f"Total: {total:.1f}M  |  "
            f"Trainable: {trainable:.1f}M  |  "
            f"Frozen: {total - trainable:.1f}M"
        )
