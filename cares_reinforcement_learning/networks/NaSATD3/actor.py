import torch
from torch import nn

from cares_reinforcement_learning.networks.NaSATD3.weight_initialization import (
    weight_init,
)


class Actor(nn.Module):
    def __init__(self, latent_size: int, num_actions: int, encoder: nn.Module):
        super().__init__()

        self.encoder_net = encoder
        self.hidden_size = [1024, 1024]

        self.act_net = nn.Sequential(
            nn.Linear(latent_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )
        self.apply(weight_init)

    def forward(
        self, state: torch.Tensor, detach_encoder: bool = False
    ) -> torch.Tensor:
        z_vector = self.encoder_net(state, detach=detach_encoder)
        output = self.act_net(z_vector)
        return output
