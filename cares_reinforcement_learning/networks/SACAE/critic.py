import torch
from torch import nn

from cares_reinforcement_learning.networks.encoders.autoencoder import Encoder


class Critic(nn.Module):
    def __init__(self, encoder: Encoder, num_actions: int):
        super().__init__()

        self.encoder = encoder

        self.hidden_size = [1024, 1024]

        # Q1 architecture
        self.Q1 = nn.Sequential(
            nn.Linear(self.encoder.latent_dim + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # Q2 architecture
        self.Q2 = nn.Sequential(
            nn.Linear(self.encoder.latent_dim + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state_latent = self.encoder(state, detach=detach_encoder)

        obs_action = torch.cat([state_latent, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
