import torch
from torch import nn

from cares_reinforcement_learning.util.configurations import DuelingDQNConfig


class DuelingNetwork(nn.Module):
    def __init__(
        self,
        observation_space_size: int,
        action_num: int,
        config: DuelingDQNConfig,
    ):
        super().__init__()

        self.hidden_size = config.hidden_size

        self.input_dim = observation_space_size
        self.output_dim = action_num

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], self.output_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
