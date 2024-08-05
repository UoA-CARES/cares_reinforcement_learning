import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder
from cares_reinforcement_learning.networks.SAC import Actor as SACActor


class Actor(SACActor):
    def __init__(
        self,
        encoder: Encoder,
        num_actions: int,
        hidden_size: list[int] = None,
        log_std_bounds: list[int] = None,
    ):
        if hidden_size is None:
            hidden_size = [1024, 1024]
        if log_std_bounds is None:
            log_std_bounds = [-10, 2]

        super().__init__(encoder.latent_dim, num_actions, hidden_size, log_std_bounds)

        self.encoder = encoder

        self.apply(hlp.weight_init)

    def forward(
        self, state: torch.Tensor, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state, detach_cnn=detach_encoder)
        return super().forward(state_latent)
