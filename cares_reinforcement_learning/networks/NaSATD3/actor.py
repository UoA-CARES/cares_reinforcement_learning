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
            output = self.autoencoder(
                state, detach_cnn=False, detach_output=detach_encoder, is_train=False
            )
            # take the mean value for stability
            z_vector = output["latent_distribution"]["mu"]
        else:
            output = self.autoencoder(
                state, detach_output=detach_encoder, is_train=False
            )
            z_vector = output["latent_observation"]

        output = self.act_net(z_vector)
        return output
