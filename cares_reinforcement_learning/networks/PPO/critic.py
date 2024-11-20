import torch
from torch import nn

from cares_reinforcement_learning.util.configurations import PPOConfig


class Critic(nn.Module):
    def __init__(self, observation_size: int, config: PPOConfig):
        super().__init__()

        self.hidden_size = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q1 = self.Q1(state)
        return q1
