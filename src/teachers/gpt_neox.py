"""TeacherAdapter for GPT-NeoX models (Pythia family)."""
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from .base import TeacherAdapter


class GPTNeoXTeacher(TeacherAdapter):
    def __init__(self, model_id: str, device: str = "cpu"):
        self._model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32)
        self._model.eval()
        self._model.to(device)
        self.device = device
        self._hooks: list = []
        self._layer_outputs: list[torch.Tensor] = []

    @property
    def model(self) -> nn.Module:
        return self._model

    @property
    def num_layers(self) -> int:
        return self._model.config.num_hidden_layers

    @property
    def hidden_size(self) -> int:
        return self._model.config.hidden_size

    def register_layer_hooks(self) -> None:
        self._layer_outputs = []
        self._hooks = []

        def make_hook(idx):
            def hook(module, input, output):
                # GPT-NeoX block output is a tuple; first element is hidden state after residual+MLP
                hidden = output[0] if isinstance(output, tuple) else output
                self._layer_outputs.append(hidden.detach())
            return hook

        for i, block in enumerate(self._model.gpt_neox.layers):
            h = block.register_forward_hook(make_hook(i))
            self._hooks.append(h)

    def remove_hooks(self) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks = []

    def get_layer_outputs(self) -> list[torch.Tensor]:
        return list(self._layer_outputs)

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        self._layer_outputs = []
        out = self._model(input_ids.to(self.device))
        return out.logits
