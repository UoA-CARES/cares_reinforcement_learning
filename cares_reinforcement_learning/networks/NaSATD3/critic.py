import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.encoders.constants import Autoencoders
from cares_reinforcement_learning.networks.encoders.autoencoder import (
    Autoencoder,
)


class Critic(nn.Module):
    def __init__(
        self,
        num_actions: int,
        autoencoder: Autoencoder,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [1024, 1024]

        self.autoencoder = autoencoder
        self.hidden_size = hidden_size

        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(self.autoencoder.latent_dim + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(self.autoencoder.latent_dim + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        self.apply(hlp.weight_init)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NaSATD3 detatches the encoder at the output
        if self.autoencoder.ae_type == Autoencoders.BURGESS:
            # take the mean value for stability
            z_vector, _, _ = self.autoencoder.encoder(
                state, detach_output=detach_encoder
            )
        else:
            z_vector = self.autoencoder.encoder(state, detach_output=detach_encoder)

        obs_action = torch.cat([z_vector, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
