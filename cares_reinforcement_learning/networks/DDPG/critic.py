import torch
from torch import nn

from cares_reinforcement_learning.util.configurations import DDPGConfig


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: DDPGConfig):
        super().__init__()

        self.hidden_size = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        return q1
