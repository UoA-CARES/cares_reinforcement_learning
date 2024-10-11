import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder


class Actor(nn.Module):
    def __init__(
        self,
        vector_observation_size: int,
        num_actions: int,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        hidden_size: list[int] | None = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [1024, 1024]

        self.num_actions = num_actions
        self.autoencoder = autoencoder
        self.hidden_size = hidden_size
        self.vector_observation_size = vector_observation_size

        self.act_net = nn.Sequential(
            nn.Linear(
                self.autoencoder.latent_dim + self.vector_observation_size,
                self.hidden_size[0],
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )
        self.apply(hlp.weight_init)

    def forward(
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> torch.Tensor:
        # NaSATD3 detatches the encoder at the output
        if self.autoencoder.ae_type == Autoencoders.BURGESS:
            # take the mean value for stability
            z_vector, _, _ = self.autoencoder.encoder(
                state["image"], detach_output=detach_encoder
            )
        else:
            z_vector = self.autoencoder.encoder(
                state["image"], detach_output=detach_encoder
            )

        actor_input = z_vector
        if self.vector_observation_size > 0:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return self.act_net(actor_input)
