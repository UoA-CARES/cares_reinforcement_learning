import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder


class Critic(nn.Module):
    def __init__(
        self,
        vector_observation_size: int,
        num_actions: int,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        hidden_size: list[int],
    ):
        super().__init__()

        self.autoencoder = autoencoder
        self.hidden_size = hidden_size
        self.vector_observation_size = vector_observation_size

        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(
                self.autoencoder.latent_dim
                + num_actions
                + self.vector_observation_size,
                self.hidden_size[0],
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(
                self.autoencoder.latent_dim
                + num_actions
                + self.vector_observation_size,
                self.hidden_size[0],
            ),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        self.apply(hlp.weight_init)

    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
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

        critic_input = z_vector
        if self.vector_observation_size > 0:
            critic_input = torch.cat([state["vector"], critic_input], dim=1)

        obs_action = torch.cat([critic_input, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
