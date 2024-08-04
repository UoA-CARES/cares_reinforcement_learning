import abc

import torch
from torch import nn

from cares_reinforcement_learning.networks.encoders.losses import (
    AELoss,
    SqVaeLoss,
    BaseBurgessLoss,
)
from cares_reinforcement_learning.networks.encoders.constants import Autoencoders


class Autoencoder(nn.Module, metaclass=abc.ABCMeta):
    """
    Base class for autoencoder models.

    Args:
        ae_type (Autoencoders): The type of autoencoder.
        loss_function (AELoss | SqVaeLoss | BaseBurgessLoss): The loss function used for training the autoencoder.
        observation_size (tuple[int]): The size of the input observations.
        latent_dim (int): The dimension of the latent space.

    Attributes:
        ae_type (Autoencoders): The type of autoencoder.
        loss_function (AELoss | SqVaeLoss | BaseBurgessLoss): The loss function used for training the autoencoder.
        observation_size (tuple[int]): The size of the input observations.
        latent_dim (int): The dimension of the latent space.
    """

    def __init__(
        self,
        ae_type: Autoencoders,
        loss_function: AELoss | SqVaeLoss | BaseBurgessLoss,
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
    ):
        """
        Forward pass of the autoencoder.

        Args:
            observation (torch.Tensor): The input observation.
            detach_cnn (bool, optional): Whether to detach the CNN part of the autoencoder. Defaults to False.
            detach_output (bool, optional): Whether to detach the output of the autoencoder. Defaults to False.
            **kwargs: Additional keyword arguments.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.

        Returns:
            torch.Tensor: The output of the autoencoder.
        """
        raise NotImplementedError("forward method must be implemented in subclass.")

    @abc.abstractmethod
    def update_autoencoder(self, data: torch.Tensor) -> float:
        """
        Update the autoencoder using the given data.

        Args:
            data (torch.Tensor): The data used for updating the autoencoder.

        Raises:
            NotImplementedError: This method must be implemented in a subclass.
        """
        raise NotImplementedError(
            "update_autoencoder method must be implemented in subclass."
        )
