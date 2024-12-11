from cares_reinforcement_learning.encoders.autoencoder_factory import AEFactory
from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder
from cares_reinforcement_learning.networks.TD3 import Actor as TD3Actor
from cares_reinforcement_learning.networks.TD3 import DefaultActor as DefaultTD3Actor
from cares_reinforcement_learning.networks.common import AEActor
from cares_reinforcement_learning.util.configurations import NaSATD3Config


class DefaultActor(AEActor):
    def __init__(self, observation_size: dict, num_actions: int):

        autoencoder = VanillaAutoencoder(
            observation_size["image"],
            latent_dim=200,
            num_layers=4,
            num_filters=32,
            kernel_size=3,
        )

        actor = DefaultTD3Actor(
            autoencoder.latent_dim, num_actions, hidden_sizes=[1024, 1024]
        )

        super().__init__(
            autoencoder,
            actor,
        )


class Actor(AEActor):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
        config: NaSATD3Config,
    ):
        ae_factory = AEFactory()
        autoencoder = ae_factory.create_autoencoder(
            observation_size=observation_size["image"], config=config.autoencoder_config
        )

        actor_observation_size = autoencoder.latent_dim
        if config.vector_observation:
            actor_observation_size += observation_size["vector"]

        actor = TD3Actor(actor_observation_size, num_actions, config)

        super().__init__(
            autoencoder,
            actor,
            add_vector_observation=bool(config.vector_observation),
        )
