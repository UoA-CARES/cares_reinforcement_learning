import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.autoencoder import (
    Autoencoder,
)


class Actor(nn.Module):
    def __init__(
        self,
        num_actions: int,
        autoencoder: Autoencoder,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [1024, 1024]

        self.num_actions = num_actions
        self.autoencoder = autoencoder
        self.hidden_size = hidden_size

        self.act_net = nn.Sequential(
            nn.Linear(self.autoencoder.latent_dim, self.hidden_size[0]),
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
        # NaSATD3 detatches the encoder at the output
        if self.autoencoder.ae_type == Autoencoders.BURGESS:
            # take the mean value for stability
            z_vector, _, _ = self.autoencoder.encoder(
                state, detach_output=detach_encoder
            )
        else:
            z_vector = self.autoencoder.encoder(state, detach_output=detach_encoder)

        output = self.act_net(z_vector)
        return output
