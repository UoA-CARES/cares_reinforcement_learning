"""BroNet residual block and initialization."""

import math

import torch
from torch import nn


class BroNetBlock(nn.Module):
    """
    BroNet residual block.

    Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm
    followed by residual addition.
    """

    def __init__(self, hidden_size: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-6),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size, eps=1e-6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.model(x)


def bro_weight_init(module: nn.Module) -> None:
    """Orthogonal initialization matching the BRO reference."""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=math.sqrt(2))

        if module.bias is not None:
            nn.init.zeros_(module.bias)
