import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder
from cares_reinforcement_learning.networks.SACD import Critic as SACDCritic

class Critic(SACDCritic):
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
            self, state: torch.Tensor, detach_encoder: bool = False
        ) -> tuple[torch.Tensor, torch.Tensor]:

        state_latent = self.encoder(state, detach_cnn=detach_encoder)
        return super().forward(state_latent)
