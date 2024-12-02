import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder
from cares_reinforcement_learning.networks.TD3 import Critic as TD3Critic
from cares_reinforcement_learning.networks.TD3 import DefaultCritic as DefaultTD3Critic
from cares_reinforcement_learning.util.configurations import NaSATD3Config


class BaseCritc(nn.Module):
    def __init__(
        self,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        critic: TD3Critic | DefaultTD3Critic,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.autoencoder = autoencoder
        self.critic = critic

        self.add_vector_observation = add_vector_observation

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
        if self.add_vector_observation:
            critic_input = torch.cat([state["vector"], critic_input], dim=1)

        return self.critic(critic_input, action)


class DefaultCritic(BaseCritc):
    def __init__(self, observation_size: dict, num_actions: int):

        autoencoder = VanillaAutoencoder(
            observation_size["image"],
            latent_dim=200,
            num_layers=4,
            num_filters=32,
            kernel_size=3,
        )

        critic = DefaultTD3Critic(
            autoencoder.latent_dim, num_actions, hidden_sizes=[1024, 1024]
        )

        super().__init__(autoencoder, critic)


class Critic(BaseCritc):
    def __init__(self, observation_size: dict, num_actions: int, config: NaSATD3Config):

        ae_factory = AEFactory()
        autoencoder = ae_factory.create_autoencoder(
            observation_size=observation_size["image"], config=config.autoencoder_config
        )

        critic_observation_size = autoencoder.latent_dim
        if config.vector_observation:
            critic_observation_size += observation_size["vector"]

        critic = TD3Critic(critic_observation_size, num_actions, config)

        super().__init__(
            autoencoder,
            critic,
            add_vector_observation=bool(config.vector_observation),
        )
