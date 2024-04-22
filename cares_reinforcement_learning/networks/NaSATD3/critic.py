import torch
from torch import nn

from cares_reinforcement_learning.networks.NaSATD3.weight_initialization import (
    weight_init,
)


class Critic(nn.Module):
    def __init__(self, latent_size: int, num_actions: int, encoder: nn.Module):
        super().__init__()

        self.encoder_net = encoder
        self.hidden_size = [1024, 1024]

        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(latent_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(latent_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        self.apply(weight_init)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        z_vector = self.encoder_net(state, detach=detach_encoder)
        obs_action = torch.cat([z_vector, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
