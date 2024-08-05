from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

# NOTE: If a parameter is a list then don't wrap with Optional leave as implicit optional - List[type] = default


class SubscriptableClass(BaseModel):
    def __getitem__(self, item):
        return getattr(self, item)


class AEConfig(SubscriptableClass):
    """
    Base Configuration class for an autoencoder.

    Attributes:
        type (str): Type of the autoencoder - vanilla or burgess.
        latent_dim (int): Dimension of the latent space.
        num_layers (Optional[int]): Number of layers in the encoder and decoder. Default is 4.
        num_filters (Optional[int]): Number of filters in each layer. Default is 32.
        kernel_size (Optional[int]): Size of the convolutional kernel. Default is 3.
    """

    type: str = Field(description="Type of the autoencoder")
    latent_dim: int
    num_layers: Optional[int] = 4
    num_filters: Optional[int] = 32
    kernel_size: Optional[int] = 3

    encoder_optim_kwargs: Optional[dict[str, float]] = Field(
        default_factory=lambda: {"lr": 1e-3}
    )

    decoder_optim_kwargs: Optional[dict[str, float]] = Field(
        default_factory=lambda: {"lr": 1e-3}
    )


class VanillaAEConfig(AEConfig):
    """
    Configuration class for the vanilla autoencoder.
    """

    type: str = "vanilla"
    latent_lambda: float = 1e-6


class SQVAEConfig(AEConfig):
    """
    Configuration class for the sqvae autoencoder.

    Attributes:

    """

    type: str = "sqvae"
    flg_arelbo: bool = Field(description="Flag to use arelbo loss function")
    loss_latent: str = Field(description="")


class BurgessConfig(AEConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
    """

    type: str = "burgess"
    rec_dist: str = Field(description="Reconstruction distribution")
    loss_function_type: str = Field(description="Loss function to use for training")


class VAEConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
    """

    loss_function_type: str = "vae"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0


class BetaHConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
    """

    loss_function_type: str = "betaH"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    beta: int = 4


class FactorKConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
    """

    loss_function_type: str = "factor"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    gamma: float = 10.0
    disc_kwargs: Optional[dict[str, float]] = Field(default_factory=lambda: None)
    optim_kwargs: Optional[dict[str, float]] = Field(
        default_factory=lambda: {"lr": 5e-5, "betas": (0.5, 0.9)}
    )


class BetaBConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
    """

    loss_function_type: str = "betaB"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    c_init: float = 0.0
    c_fin: float = 20.0
    gamma: float = 100.0


class BTCVAEConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
    """

    loss_function_type: str = "btcVAE"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    alpha: float = 1.0
    beta: float = 6.0
    gamma: float = 1.0
    is_mss: bool = True
