from typing import Optional

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

    Attributes:
        type (str): The type of autoencoder. Default is "vanilla".
        latent_lambda (float): The value of the latent lambda. Default is 1e-6.
    """

    type: str = "vanilla"
    latent_lambda: float = 1e-6


# sqVAE = parser.add_argument_group('SQ-VAE specific parameters')
# sqVAE.add_argument('--dim_z', type=int, default=16)
# sqVAE.add_argument('--size_dict', type=int, default=512)
# sqVAE.add_argument('--param_var_q', type=str, default=ParamVarQ.GAUSSIAN_1.value,
#                     choices=[pvq.value for pvq in ParamVarQ])
# sqVAE.add_argument('--num_rb', type=int, default=6)
# sqVAE.add_argument('--flg_arelbo', type=bool, default=True)
# sqVAE.add_argument('--log_param_q_init', type=float, default=0.0)
# sqVAE.add_argument('--temperature_init', type=float, default=1.0)

# class SQVAEConfig(AEConfig):
#     """
#     Configuration class for the sqvae autoencoder.

#     Attributes:

#     """

#     type: str = "sqvae"
#     flg_arelbo: bool = Field(description="Flag to use arelbo loss function")
#     loss_latent: str = Field(description="")


class BurgessConfig(AEConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        type (str): Type of the autoencoder.
        rec_dist (str): Reconstruction distribution.
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
        rec_dist (str): Reconstruction distribution type.
        steps_anneal (int): Number of steps for annealing.
    """

    loss_function_type: str = "vae"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0


class BetaHConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
        rec_dist (str): Reconstruction distribution.
        steps_anneal (int): Number of steps to anneal the loss function.
        beta (int): Beta value for the loss function.
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
        rec_dist (str): Reconstruction distribution type.
        steps_anneal (int): Number of steps to anneal the loss function.
        gamma (float): Gamma value for the loss function.
        disc_kwargs (Optional[dict[str, float]]): Keyword arguments for the discriminator.
        optim_kwargs (Optional[dict[str, float]]): Keyword arguments for the optimizer.
    """

    loss_function_type: str = "factor"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    gamma: float = 6.0
    disc_kwargs: Optional[dict[str, float]] = Field(default_factory=lambda: None)
    optim_kwargs: Optional[dict[str, float]] = Field(
        default_factory=lambda: {"lr": 5e-5, "betas": (0.5, 0.9)}
    )


class BetaBConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
        rec_dist (str): Reconstruction distribution type.
        steps_anneal (int): Number of steps for annealing.
        c_init (float): Initial value for the capacity.
        c_fin (float): Final value for the capacity.
        gamma (float): Gamma value for the capacity.
    """

    loss_function_type: str = "betaB"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    c_init: float = 0.0
    c_fin: float = 25.0
    gamma: float = 100.0


class BTCVAEConfig(BurgessConfig):
    """
    Configuration class for the Burgess autoencoder.

    Attributes:
        loss_function_type (str): Loss function to use for training.
        rec_dist (str): Reconstruction distribution type.
        steps_anneal (int): Number of steps to anneal the loss function.
        alpha (float): Alpha parameter for the loss function.
        beta (float): Beta parameter for the loss function.
        gamma (float): Gamma parameter for the loss function.
        is_mss (bool): Flag indicating whether to use the Mean-Subtract-Square (MSS) trick.
    """

    loss_function_type: str = "btcVAE"
    rec_dist: str = "bernoulli"
    steps_anneal: int = 0
    alpha: float = 1.0
    beta: float = 6.0
    gamma: float = 1.0
    is_mss: bool = True
