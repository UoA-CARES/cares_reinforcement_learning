import torch
from torch import nn


class SIMBABlock(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        return input_value + self.model(input_value)
