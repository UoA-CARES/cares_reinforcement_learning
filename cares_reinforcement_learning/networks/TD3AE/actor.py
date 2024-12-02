import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder
from cares_reinforcement_learning.networks.TD3 import Actor as TD3Actor
from cares_reinforcement_learning.networks.TD3 import DefaultActor as DefaultTD3Actor
from cares_reinforcement_learning.util.configurations import TD3AEConfig


class BaseActor(nn.Module):
    def __init__(
        self,
        num_actions: int,
        encoder: Encoder,
        actor: TD3Actor | DefaultTD3Actor,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.encoder = encoder
        self.actor = actor

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(  # type: ignore
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        actor_input = state_latent
        if self.add_vector_observation:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return self.actor(actor_input)


class DefaultActor(BaseActor):
    def __init__(self, observation_size: dict, num_actions: int):

        encoder = Encoder(
            observation_size["image"],
            latent_dim=50,
            num_layers=4,
            num_filters=32,
            kernel_size=3,
        )

        actor = DefaultTD3Actor(
            encoder.latent_dim, num_actions, hidden_sizes=[1024, 1024]
        )

        super().__init__(
            num_actions,
            encoder,
            actor,
        )


class Actor(BaseActor):
    def __init__(self, observation_size: dict, num_actions: int, config: TD3AEConfig):

        ae_config = config.autoencoder_config
        encoder = Encoder(
            observation_size["image"],
            latent_dim=ae_config.latent_dim,
            num_layers=ae_config.num_layers,
            num_filters=ae_config.num_filters,
            kernel_size=ae_config.kernel_size,
        )

        actor_observation_size = encoder.latent_dim
        if config.vector_observation:
            actor_observation_size += observation_size["vector"]

        actor = TD3Actor(actor_observation_size, num_actions, config)

        super().__init__(
            num_actions,
            encoder=encoder,
            actor=actor,
            add_vector_observation=bool(config.vector_observation),
        )
