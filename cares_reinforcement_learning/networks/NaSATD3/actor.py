import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.encoders.constants import Autoencoders


class Actor(nn.Module):
    def __init__(
        self,
        latent_size: int,
        num_actions: int,
        autoencoder: nn.Module,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [1024, 1024]

        self.autoencoder = autoencoder
        self.hidden_size = hidden_size

        self.act_net = nn.Sequential(
            nn.Linear(latent_size, self.hidden_size[0]),
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
