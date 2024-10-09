import logging

from cares_reinforcement_learning.encoders import losses
from cares_reinforcement_learning.encoders.autoencoder import Autoencoder
from cares_reinforcement_learning.encoders.configurations import AEConfig

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


def create_vanilla_autoencoder(
    observation_size: tuple[int],
    config: AEConfig,
) -> Autoencoder:
    from cares_reinforcement_learning.encoders.vanilla_autoencoder import (
        VanillaAutoencoder,
    )

    return VanillaAutoencoder(
        observation_size=observation_size,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        latent_lambda=config.latent_lambda,
        encoder_optimiser_params=config.encoder_optim_kwargs,
        decoder_optimiser_params=config.decoder_optim_kwargs,
    )


def create_burgess_autoencoder(
    observation_size: tuple[int],
    config: AEConfig,
) -> Autoencoder:
    from cares_reinforcement_learning.encoders.burgess_autoencoder import (
        BurgessAutoencoder,
    )

    loss_function = losses.get_burgess_loss_function(config)

    return BurgessAutoencoder(
        loss_function=loss_function,
        observation_size=observation_size,
        latent_dim=config.latent_dim,
        num_layers=config.num_layers,
        num_filters=config.num_filters,
        kernel_size=config.kernel_size,
        encoder_optimiser_params=config.encoder_optim_kwargs,
        decoder_optimiser_params=config.decoder_optim_kwargs,
    )


class AEFactory:
    def create_autoencoder(
        self,
        observation_size: tuple[int],
        config: AEConfig,
    ) -> Autoencoder:

        autoencoder = None
        if config.type == "vanilla":
            autoencoder = create_vanilla_autoencoder(observation_size, config)
        elif config.type == "burgess":
            autoencoder = create_burgess_autoencoder(observation_size, config)

        if autoencoder is None:
            logging.warning(f"Unkown autoencoder {autoencoder}.")

        return autoencoder
