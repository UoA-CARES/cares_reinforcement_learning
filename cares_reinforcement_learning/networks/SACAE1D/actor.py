from cares_reinforcement_learning.encoders.vanilla_autoencoder import Encoder1D
from cares_reinforcement_learning.networks.SAC import Actor as SACActor
from cares_reinforcement_learning.networks.SAC import DefaultActor as DefaultSACActor
from cares_reinforcement_learning.networks.common import EncoderPolicy1D
from cares_reinforcement_learning.util.configurations import SACAE1DConfig


class DefaultActor(EncoderPolicy1D):
    def __init__(self, observation_size: int | dict, num_actions: int):

        if isinstance(observation_size, dict):
            # observation size dict should be {"lidar": int, "vector": int}
            encoder = Encoder1D(
                observation_size["lidar"],
                latent_dim=50,
                num_layers=4,
                num_filters=32,
                kernel_size=3,
            )
        else:
            encoder = Encoder1D(
                observation_size,
                latent_dim=50,
                num_layers=4,
                num_filters=32,
                kernel_size=3,
            )

        actor = DefaultSACActor(
            encoder.latent_dim, num_actions, hidden_sizes=[1024, 1024]
        )

        super().__init__(
            encoder,
            actor,
        )


class Actor(EncoderPolicy1D):
    def __init__(self, observation_size: int | dict, num_actions: int, config: SACAE1DConfig):

        ae_config = config.autoencoder_config
        if isinstance(observation_size, dict):
            encoder = Encoder1D(
                observation_size["lidar"],
                latent_dim=ae_config.latent_dim,
                num_layers=ae_config.num_layers,
                num_filters=ae_config.num_filters,
                kernel_size=ae_config.kernel_size,
            )
        else:
            encoder = Encoder1D(
                observation_size,
                latent_dim=ae_config.latent_dim,
                num_layers=ae_config.num_layers,
                num_filters=ae_config.num_filters,
                kernel_size=ae_config.kernel_size,
            )

        actor_observation_size = encoder.latent_dim
        if config.vector_observation:
            actor_observation_size += observation_size["vector"]

        actor = SACActor(actor_observation_size, num_actions, config)

        super().__init__(
            encoder=encoder,
            actor=actor,
            add_vector_observation=bool(config.vector_observation),
        )
