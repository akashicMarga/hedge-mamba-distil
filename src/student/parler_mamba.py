"""ParlerMamba: HF ParlerTTS decoder with HedgeMamba SSM self-attention.

The HuggingFace parler_tts library provides ParlerTTSForConditionalGeneration.
This module wraps it: each decoder self-attention layer is replaced with
ParlerHedgeMambaLayer (a HedgeMambaMixer with the Whisper-style interface).

Weight surgery follows Appendix B of arXiv:2604.14191:
  B_proj (hhog_k.phi) <- teacher self_attn.k_proj
  C_proj (hhog_q.phi) <- teacher self_attn.q_proj
  v_proj              <- teacher self_attn.v_proj
  out_proj            <- teacher self_attn.out_proj

All other decoder weights (cross-attn, FFN, LN, embeddings, T5 encoder)
are copied unchanged from the teacher.

Distillation stages are in src/distill/parler_stage{1,2}.py.
"""
import copy
import torch
import torch.nn as nn
from .hedge_mamba import HedgeMambaMixer


class ParlerHedgeMambaLayer(nn.Module):
    """Drop-in for HF ParlerTTSAttention inside a decoder self-attn slot.

    Wraps HedgeMambaMixer and handles:
      - HF decoder interface: (hidden_states, **kwargs) -> (out, None, None)
      - Fix-B caching: _h_cache / _conv_cache for O(1) AR generation
    """

    def __init__(self, dim: int, num_heads: int,
                 state_size: int | None = None):
        super().__init__()
        self.mixer = HedgeMambaMixer(dim=dim, num_heads=num_heads,
                                     state_size=state_size)
        self._h_cache:    torch.Tensor | None = None
        self._conv_cache: torch.Tensor | None = None
        self._caching_active: bool = False

    def reset_ssm_cache(self) -> None:
        self._h_cache    = None
        self._conv_cache = None

    def forward(self, hidden_states: torch.Tensor, **kwargs):
        # Accept and ignore all HF attention kwargs
        # (past_key_value, attention_mask, layer_head_mask, …)
        h_prev    = self._h_cache    if self._caching_active else None
        conv_prev = self._conv_cache if self._caching_active else None

        out, h_new, conv_new = self.mixer(
            hidden_states, h_prev=h_prev, conv_cache=conv_prev
        )
        if self._caching_active:
            self._h_cache    = h_new.detach()
            self._conv_cache = conv_new.detach()
        # HF attention expects (attn_output, past_key_value, attn_weights)
        return out, None, None


class ParlerMambaStudent(nn.Module):
    """Full ParlerTTS-Mamba student.

    Args:
        parler_model:  ParlerTTSForConditionalGeneration (HF)
        state_size:    SSM state per channel before Hedgehog doubling.
                       None → hidden_size (paper default, N=D).

    Usage::

        from parler_tts import ParlerTTSForConditionalGeneration
        teacher = ParlerTTSForConditionalGeneration.from_pretrained(
            "ai4bharat/indic-parler-tts"
        )
        student = ParlerMambaStudent(teacher)
        student = student.to(device)

    Forward::

        out = student(
            input_ids=description_ids,
            decoder_input_ids=decoder_input_ids,
        )
        # out.logits: (B, T, num_codebooks * vocab_size)
    """

    def __init__(self, parler_model, state_size: int | None = None):
        super().__init__()
        self.backbone = copy.deepcopy(parler_model)
        self.config   = parler_model.config

        # Replace each decoder self-attention with HedgeMambaMixer
        decoder_layers = self.backbone.decoder.model.decoder.layers
        for layer in decoder_layers:
            dim       = layer.self_attn.q_proj.in_features
            num_heads = layer.self_attn.num_heads
            layer.self_attn = ParlerHedgeMambaLayer(
                dim=dim, num_heads=num_heads, state_size=state_size
            )

        # Freeze T5 encoder entirely
        for p in self.backbone.text_encoder.parameters():
            p.requires_grad = False

        # Freeze prompt embeddings (shared vocab)
        for p in self.backbone.embed_prompts.parameters():
            p.requires_grad = False

        # Freeze audio token embeddings and position embeddings
        dec = self.backbone.decoder.model.decoder
        for emb in dec.embed_tokens:
            for p in emb.parameters():
                p.requires_grad = False
        for p in dec.embed_positions.parameters():
            p.requires_grad = False

    # -- surgery --------------------------------------------------------------

    @classmethod
    def from_pretrained_with_surgery(
        cls,
        repo_id: str = "ai4bharat/indic-parler-tts",
        state_size: int | None = None,
        device: str = "cpu",
    ) -> "ParlerMambaStudent":
        """Load teacher, build student, warm-start SSM from attn weights."""
        from parler_tts import ParlerTTSForConditionalGeneration
        teacher = ParlerTTSForConditionalGeneration.from_pretrained(repo_id)
        teacher = teacher.to(device)
        student = cls(teacher, state_size=state_size)
        student._apply_param_surgery(teacher)
        student = student.to(device)
        return student

    def _apply_param_surgery(self, teacher_model) -> None:
        """Warm-start SSM B/C/v/out from teacher self-attn projections.

        Maps (paper Appendix B):
          hhog_k.phi.weight <- teacher.k_proj.weight  (keys  → state input)
          hhog_q.phi.weight <- teacher.q_proj.weight  (queries → state out)
          v_proj.weight     <- teacher.v_proj.weight
          out_proj.weight   <- teacher.out_proj.weight
        """
        teacher_layers = teacher_model.decoder.model.decoder.layers
        student_layers = self.backbone.decoder.model.decoder.layers
        for t_layer, s_layer in zip(teacher_layers, student_layers):
            t_attn = t_layer.self_attn          # original ParlerTTSAttention
            s_ssm  = s_layer.self_attn.mixer    # HedgeMambaMixer
            N      = s_ssm.state_size

            with torch.no_grad():
                # Keys -> B projection (Hedgehog phi weight)
                s_ssm.hhog_k.phi.weight.copy_(t_attn.k_proj.weight[:N, :])
                # Queries -> C projection
                s_ssm.hhog_q.phi.weight.copy_(t_attn.q_proj.weight[:N, :])
                # Value and output projections map directly
                s_ssm.v_proj.weight.copy_(t_attn.v_proj.weight)
                s_ssm.out_proj.weight.copy_(t_attn.out_proj.weight)
                # Conv: near-identity (last kernel position = 1.0)
                s_ssm.conv1d.weight.zero_()
                s_ssm.conv1d.weight[:, :, -1] = 1.0

    # -- Fix-B helpers --------------------------------------------------------

    def _set_ssm_caching(self, active: bool) -> None:
        for layer in self.backbone.decoder.model.decoder.layers:
            if isinstance(layer.self_attn, ParlerHedgeMambaLayer):
                layer.self_attn._caching_active = active
                layer.self_attn.reset_ssm_cache()

    # -- forward --------------------------------------------------------------

    def forward(self, **kwargs):
        return self.backbone(**kwargs)

    def generate(self, **kwargs):
        """Generate with Fix-B SSM state caching."""
        self._set_ssm_caching(True)
        try:
            return self.backbone.generate(**kwargs)
        finally:
            self._set_ssm_caching(False)

    # -- utilities ------------------------------------------------------------

    def trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def param_summary(self) -> str:
        total     = sum(p.numel() for p in self.parameters()) / 1e6
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad) / 1e6
        return (
            f"Total: {total:.1f}M  |  "
            f"Trainable: {trainable:.1f}M  |  "
            f"Frozen: {total - trainable:.1f}M"
        )

    def export_mamba_weights(self, path: str) -> None:
        """Export Mamba SSM layer weights to a flat .npz for MLX loading.

        The saved keys follow the HF-style naming that mamba_model.py's
        _remap_mamba_weights() expects:
          decoder.model.decoder.layers.{i}.self_attn.{param}
        """
        import numpy as np
        out = {}
        layers = self.backbone.decoder.model.decoder.layers
        for i, layer in enumerate(layers):
            ssm = layer.self_attn.mixer
            prefix = f"decoder.model.decoder.layers.{i}.self_attn"
            out[f"{prefix}.gate_proj.weight"]  = ssm.gate_proj.weight.detach().cpu().numpy()
            out[f"{prefix}.x_proj.weight"]     = ssm.x_proj.weight.detach().cpu().numpy()
            out[f"{prefix}.dt_proj.weight"]    = ssm.dt_proj.weight.detach().cpu().numpy()
            out[f"{prefix}.dt_proj.bias"]      = ssm.dt_proj.bias.detach().cpu().numpy()
            out[f"{prefix}.A_log"]             = ssm.A_log.detach().cpu().numpy()
            out[f"{prefix}.v_proj.weight"]     = ssm.v_proj.weight.detach().cpu().numpy()
            out[f"{prefix}.out_proj.weight"]   = ssm.out_proj.weight.detach().cpu().numpy()
            out[f"{prefix}.hhog_k.phi.weight"] = ssm.hhog_k.phi.weight.detach().cpu().numpy()
            out[f"{prefix}.hhog_q.phi.weight"] = ssm.hhog_q.phi.weight.detach().cpu().numpy()
            # Conv1d: (D,1,k) in PT -> store as-is; MLX loader transposes
            out[f"{prefix}.conv_weight"] = ssm.conv1d.weight.detach().cpu().numpy()
        np.savez(path, **out)
        print(f"[export] saved {len(out)} tensors to {path}")
