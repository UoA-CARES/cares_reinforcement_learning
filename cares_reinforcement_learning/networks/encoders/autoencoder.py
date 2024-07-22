import torch
from torch import nn


class Autoencoder(nn.Module):
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
        loss_function,
        observation_size: tuple[int],
        latent_dim: int,
    ):
        super().__init__()

        self.loss_function = loss_function
        self.observation_size = observation_size
        self.latent_dim = latent_dim

    def forward(
        self,
        observation: torch.Tensor,
        detach_cnn: bool = False,
        detach_output: bool = False,
        **kwargs,
    ):
        raise NotImplementedError("forward method must be implemented in subclass.")
