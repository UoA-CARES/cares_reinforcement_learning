import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.encoders.autoencoder import Encoder


class Actor(nn.Module):
    def __init__(self, encoder: Encoder, num_actions: int):
        super().__init__()

        self.encoder = encoder

        self.hidden_size = [1024, 1024]

        self.act_net = nn.Sequential(
            nn.Linear(self.encoder.latent_dim, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )

        self.apply(hlp.weight_init)

    def forward(
        self, state: torch.Tensor, detach_encoder: bool = False
    ) -> torch.Tensor:
        state_latent = self.encoder(state, detach=detach_encoder)
        output = self.act_net(state_latent)
        return output
