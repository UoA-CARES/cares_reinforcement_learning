import torch
from torch import nn

from cares_reinforcement_learning.util.configurations import DDPGConfig


class Actor(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: DDPGConfig):
        super().__init__()

        self.num_actions = num_actions
        self.hidden_size = config.hidden_size_actor

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output
