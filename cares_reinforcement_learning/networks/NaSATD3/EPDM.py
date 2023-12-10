"""
Ensemble of Predictive Discrete Model (EPPM)
Predict outputs  a point estimate e.g. discrete value
"""

import torch
from torch import nn

from cares_reinforcement_learning.networks.NaSATD3.weight_initialization import (
    weight_init,
)


# pylint: disable-next=invalid-name
class EPDM(nn.Module):
    def __init__(self, latent_size, num_actions):
        super().__init__()

        self.input_dim = latent_size + num_actions
        self.output_dim = latent_size
        self.hidden_size = [512, 512]

        self.prediction_net = nn.Sequential(
            nn.Linear(in_features=self.input_dim, out_features=self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(
                in_features=self.hidden_size[0], out_features=self.hidden_size[1]
            ),
            nn.ReLU(),
            nn.Linear(in_features=self.hidden_size[1], out_features=self.output_dim),
        )

        self.apply(weight_init)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        out = self.prediction_net(x)
        return out
