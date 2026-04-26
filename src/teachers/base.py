from abc import ABC, abstractmethod
from typing import Iterator
import torch
import torch.nn as nn


class TeacherAdapter(ABC):
    """ABC for wrapping teacher models to expose per-layer outputs for distillation."""

    @property
    @abstractmethod
    def model(self) -> nn.Module:
        ...

    @property
    @abstractmethod
    def num_layers(self) -> int:
        ...

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        ...

    @abstractmethod
    def register_layer_hooks(self) -> None:
        """Register forward hooks to capture per-layer outputs after residual add."""
        ...

    @abstractmethod
    def remove_hooks(self) -> None:
        ...

    @abstractmethod
    def get_layer_outputs(self) -> list[torch.Tensor]:
        """Return captured outputs from last forward pass, one per layer."""
        ...

    @abstractmethod
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run teacher forward, populating layer output cache."""
        ...

    def parameters(self) -> Iterator[nn.Parameter]:
        return self.model.parameters()

    def eval(self) -> "TeacherAdapter":
        self.model.eval()
        return self

    def to(self, device) -> "TeacherAdapter":
        self.model.to(device)
        return self
