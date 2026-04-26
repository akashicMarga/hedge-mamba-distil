"""Build the student model by replacing GPTNeoXAttention blocks with HedgeMambaLayer."""
import copy
import torch.nn as nn
from transformers import AutoModelForCausalLM
from .hedge_mamba import HedgeMambaLayer


class HedgeMambaStudent(nn.Module):
    def __init__(self, teacher_model: nn.Module, stage: str = "linear"):
        super().__init__()
        self.config = teacher_model.config
        # Deep copy to avoid mutating teacher
        self.backbone = copy.deepcopy(teacher_model)
        self.stage = stage
        self._replace_attention_layers(stage)

    def _replace_attention_layers(self, stage: str) -> None:
        cfg = self.config
        dim = cfg.hidden_size
        num_heads = cfg.num_attention_heads

        for block in self.backbone.gpt_neox.layers:
            # Replace the attention sub-module; keep MLP, LayerNorm, residual intact
            block.attention = HedgeMambaLayer(dim, num_heads, stage=stage)

    def forward(self, input_ids, **kwargs):
        return self.backbone(input_ids, **kwargs)

    def upgrade_to_ssm(self) -> None:
        """In-place Stage 1→2 upgrade using Appendix B parameter surgery."""
        from .param_init import apply_surgery
        cfg = self.config
        dim = cfg.hidden_size
        num_heads = cfg.num_attention_heads

        for block in self.backbone.gpt_neox.layers:
            if isinstance(block.attention, HedgeMambaLayer) and block.attention.stage == "linear":
                block.attention = apply_surgery(block.attention)
        self.stage = "ssm"
