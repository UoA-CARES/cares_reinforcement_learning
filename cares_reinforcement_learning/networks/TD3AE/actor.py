import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder
from cares_reinforcement_learning.networks.TD3 import Actor as TD3Actor


class Actor(TD3Actor):
    def __init__(
        self,
        vector_observation_size: int,
        encoder: Encoder,
        num_actions: int,
        hidden_size: list[int] | None = None,
    ):
        if hidden_size is None:
            hidden_size = [1024, 1024]

        super().__init__(
            encoder.latent_dim + vector_observation_size, num_actions, hidden_size
        )

        self.encoder = encoder

        self.apply(hlp.weight_init)

        self.vector_observation_size = vector_observation_size

    def forward(
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> torch.Tensor:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        actor_input = state_latent
        if self.vector_observation_size > 0:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return super().forward(actor_input)
