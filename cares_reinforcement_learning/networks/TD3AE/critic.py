import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.encoders.autoencoder import Encoder
from cares_reinforcement_learning.networks.TD3 import Critic as TD3Critic


class Critic(TD3Critic):
    def __init__(
        self,
        encoder: Encoder,
        num_actions: int,
        hidden_size: list[int] = None,
    ):
        if hidden_size is None:
            hidden_size = [1024, 1024]

        super().__init__(encoder.latent_dim, num_actions, hidden_size)

        self.encoder = encoder

        self.apply(hlp.weight_init)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state, detach_cnn=detach_encoder)
        return super().forward(state_latent, action)
