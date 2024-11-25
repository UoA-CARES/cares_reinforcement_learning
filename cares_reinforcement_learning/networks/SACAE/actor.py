import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder
from cares_reinforcement_learning.networks.SAC import DefaultActor as DefaultSACActor
from cares_reinforcement_learning.networks.SAC import Actor as SACActor
from cares_reinforcement_learning.util.configurations import SACAEConfig


# class BaseActor(DefaultSACActor):
#     def __init__(self, observation_size: int, num_actions: int):
#         pass


class BaseActor:
    def __init__(self, encoder: Encoder, actor: SACActor | DefaultSACActor):
        super().__init__()

        self.encoder = encoder
        self.actor = actor

        self.add_vector_observation = False

    def forward(  # type: ignore
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        actor_input = state_latent
        if self.add_vector_observation:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return self.actor(actor_input)


class Actor(SACActor):
    def __init__(
        self,
        vector_observation_size: int,
        encoder: Encoder,
        num_actions: int,
        config: SACAEConfig,
    ):
        super().__init__(
            encoder.latent_dim + vector_observation_size, num_actions, config
        )

        self.encoder = encoder

        self.vector_observation_size = vector_observation_size

        self.apply(hlp.weight_init)

    def forward(  # type: ignore
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        actor_input = state_latent
        if self.vector_observation_size > 0:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return super().forward(actor_input)
