"""
Ensemble of Predictive Discrete Model (EPDM)
Predict outputs  a point estimate e.g. discrete value
"""

import torch
from torch import nn

from cares_reinforcement_learning.networks.CORETD3.weight_initialization import (
    weight_init,
)


# pylint: disable-next=invalid-name
class EPDM(nn.Module):
    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()

        self.hidden_size = [256, 256]

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )
        self.apply(weight_init)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output

    
