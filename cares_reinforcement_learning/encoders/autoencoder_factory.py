from typing import overload

import cares_reinforcement_learning.encoders.configurations as acf
from cares_reinforcement_learning.encoders import losses
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder

# Disable these as this is a deliberate use of dynamic imports
# pylint: disable=import-outside-toplevel
# pylint: disable=invalid-name


def create_vanilla_autoencoder(
    observation_size: tuple[int],
    config: acf.VanillaAEConfig,
) -> VanillaAutoencoder:

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
    config: acf.BurgessConfig,
) -> BurgessAutoencoder:

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
    @overload
    def create_autoencoder(
        self,
        observation_size: tuple[int],
        config: acf.VanillaAEConfig,
    ) -> VanillaAutoencoder: ...

    @overload
    def create_autoencoder(
        self,
        observation_size: tuple[int],
        config: acf.BurgessConfig,
    ) -> BurgessAutoencoder: ...

    def create_autoencoder(
        self,
        observation_size: tuple[int],
        config: acf.VanillaAEConfig | acf.BurgessConfig,
    ) -> BurgessAutoencoder | VanillaAutoencoder:

        if isinstance(config, acf.VanillaAEConfig):
            return create_vanilla_autoencoder(observation_size, config)

        if isinstance(config, acf.BurgessConfig):
            return create_burgess_autoencoder(observation_size, config)

        raise ValueError(
            f"Invalid autoencoder configuration: {type(config)=} {config=}"
        )
