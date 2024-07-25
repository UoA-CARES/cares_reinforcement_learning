import abc

import torch
from torch import nn

from cares_reinforcement_learning.networks.encoders.losses import BaseLoss
from cares_reinforcement_learning.networks.encoders.constants import Autoencoders


class Autoencoder(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for the autoencoder models consisting of an encoder and a decoder pair.

    Args:
        observation_size (tuple[int]): The size of the input image observations.
        latent_dim (int): The dimension of the latent space.
        num_layers (int, optional): The number of layers in the encoder and decoder. Defaults to 4.
        num_filters (int, optional): The number of filters in each layer. Defaults to 32.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
    """

    def __init__(
        self,
        ae_type: Autoencoders,
        loss_function: BaseLoss,
        observation_size: tuple[int],
        latent_dim: int,
    ):
        super().__init__()

        self.ae_type = ae_type
        self.loss_function = loss_function
        self.observation_size = observation_size
        self.latent_dim = latent_dim

    @abc.abstractmethod
    def forward(
        self,
        observation: torch.Tensor,
        detach_cnn: bool = False,
        detach_output: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("Forward method must be implemented in subclass.")
