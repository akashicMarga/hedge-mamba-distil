"""WhisperMambaMLX — full model for MLX training.

Wraps mlx-community/whisper-tiny-mlx:
  encoder:     frozen AudioEncoder (untouched)
  decoder:     TextDecoder with self-attention replaced by HedgeMambaMixerMLX
               cross-attn, FFN, layer norms: trainable in Stage 2

Forward returns (logits, hidden_states) where hidden_states is a list of
per-block outputs — used for Stage 1 cosine distillation.
"""
import mlx.core as mx
import mlx.nn as nn
import mlx_whisper.load_models as lm
from mlx.utils import tree_map, tree_flatten

from src.student.mlx_hedge_mamba import HedgeMambaMixerMLX


def _cast_to_float32(model) -> None:
    """Cast all parameters and non-parameter buffers to float32.

    MLX loads the community model in float16.  Training needs float32 to avoid
    silent precision loss (same gotcha as MPS — see CLAUDE.md).
    """
    params = tree_map(
        lambda x: x.astype(mx.float32) if isinstance(x, mx.array) else x,
        model.parameters(),
    )
    model.update(params)
    # Underscore-prefixed attributes are not in parameters() — cast separately.
    model.decoder._mask = model.decoder._mask.astype(mx.float32)
    model.encoder._positional_embedding = (
        model.encoder._positional_embedding.astype(mx.float32)
    )
    mx.eval(model.parameters())


def get_teacher_hiddens(
    teacher, mel: mx.array, decoder_input_ids: mx.array
) -> list[mx.array]:
    """Run unpatched Whisper decoder, collecting post-block hidden states.

    Used in Stage 1: teacher is fully frozen, called outside value_and_grad.
    """
    B, L = decoder_input_ids.shape
    xa = teacher.encoder(mel)
    x = (
        teacher.decoder.token_embedding(decoder_input_ids)
        + teacher.decoder.positional_embedding[:L]
    )
    mask = teacher.decoder._mask
    hidden_states = []
    for block in teacher.decoder.blocks:
        x, _, _ = block(x, xa, mask=mask, kv_cache=None)
        hidden_states.append(x)
    return hidden_states


class WhisperMambaMLX(nn.Module):
    """Whisper-tiny backbone with SSM decoder, ready for MLX training.

    Usage:
        student = WhisperMambaMLX(state_size=64)
        student.freeze_for_stage1()          # only SSM weights train
        logits, hiddens = student(mel, ids)  # mel: (B, T=3000, 80)
    """

    def __init__(
        self,
        mlx_repo: str = "mlx-community/whisper-tiny-mlx",
        state_size: int = 64,
    ):
        super().__init__()
        self.model = lm.load_model(mlx_repo)
        self.state_size = state_size
        dims = self.model.dims

        # Cast to float32 before patching so existing cross-attn/FFN weights land in float32
        _cast_to_float32(self.model)

        # Replace each decoder self-attention with HedgeMambaMixerMLX
        for block in self.model.decoder.blocks:
            block.attn = HedgeMambaMixerMLX(
                dim=dims.n_text_state,
                num_heads=dims.n_text_head,
                state_size=state_size,
            )

        # Default freeze: Stage 2 (encoder + embeddings frozen, decoder trains)
        self.freeze_for_stage2()

    # ── Freeze helpers ────────────────────────────────────────────────────────

    def freeze_for_stage1(self) -> None:
        """Stage 1: only SSM layers (block.attn) are trainable."""
        self.model.freeze()
        for block in self.model.decoder.blocks:
            block.attn.unfreeze()

    def freeze_for_stage2(self) -> None:
        """Stage 2: decoder blocks + final LN train; encoder + embeddings frozen.

        positional_embedding and token_embedding remain frozen because we only
        unfreeze specific sub-modules (blocks and decoder.ln), not the decoder
        module itself.
        """
        self.model.freeze()  # freeze everything
        for block in self.model.decoder.blocks:
            block.unfreeze()
        self.model.decoder.ln.unfreeze()
        # token_embedding is a sub-module on decoder — re-freeze (already frozen,
        # but explicit for clarity since block.unfreeze() could cascade upward).
        self.model.decoder.token_embedding.freeze()

    # ── Forward ───────────────────────────────────────────────────────────────

    def encode(self, mel: mx.array) -> mx.array:
        """mel: (B, T=3000, 80) → (B, 1500, D)"""
        return self.model.encoder(mel)

    def __call__(
        self, mel: mx.array, decoder_input_ids: mx.array
    ) -> tuple[mx.array, list[mx.array]]:
        """
        mel:               (B, T=3000, 80)
        decoder_input_ids: (B, L)  — int32

        Returns:
            logits:        (B, L, vocab)
            hidden_states: list[(B, L, D)], one per decoder block
        """
        B, L = decoder_input_ids.shape

        xa = self.model.encoder(mel)  # (B, 1500, D)  — frozen encoder

        x = (
            self.model.decoder.token_embedding(decoder_input_ids)
            + self.model.decoder.positional_embedding[:L]
        )

        # Reuse the pre-built causal mask from TextDecoder (underscore = buffer, not param)
        mask = self.model.decoder._mask

        hidden_states = []
        for block in self.model.decoder.blocks:
            x, _, _ = block(x, xa, mask=mask, kv_cache=None)
            hidden_states.append(x)

        x = self.model.decoder.ln(x)
        logits = self.model.decoder.token_embedding.as_linear(x)  # (B, L, vocab)

        return logits, hidden_states

    # ── Utilities ─────────────────────────────────────────────────────────────

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
