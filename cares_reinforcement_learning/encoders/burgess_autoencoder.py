"""
The Burgess Autoencoder implementations and the variations of loss functions has been sourced from the link below.

We have adapted their work into a generalised form for our use case in RL.

Original Code: https://github.com/YannDubs/disentangling-vae/tree/master
"""

from typing import Any

import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.losses import BaseBurgessLoss


def tie_weights(src, trg):
    trg.weight = src.weight
    trg.bias = src.bias


class BurgessAutoencoder(nn.Module):
    """
    Implementation of the Burgess Autoencoder.

    Args:
        observation_size (tuple[int]): The size of the input observation.
        latent_dim (int): The dimension of the latent space.
        loss_function (BaseBurgessLoss): The loss function used for training the autoencoder.
        num_layers (int, optional): The number of layers in the encoder and decoder networks. Defaults to 4.
        num_filters (int, optional): The number of filters in each layer of the encoder and decoder networks. Defaults to 32.
        kernel_size (int, optional): The size of the kernel used in the convolutional layers of the encoder and decoder networks. Defaults to 3.
        encoder_optimiser_params (dict[str, any], optional): Additional parameters for the encoder optimizer. Defaults to None.
        decoder_optimiser_params (dict[str, any], optional): Additional parameters for the decoder optimizer. Defaults to None.

    Attributes:
        encoder (BurgessEncoder): The encoder network.
        decoder (BurgessDecoder): The decoder network.
        encoder_optimizer (torch.optim.Adam): The optimizer for the encoder network.
        decoder_optimizer (torch.optim.Adam): The optimizer for the decoder network.

    Methods:
        update_autoencoder(data: torch.Tensor) -> None:
            Update the autoencoder parameters based on the given data.
        forward(observation, detach_cnn: bool = False, detach_output: bool = False, **kwargs) -> dict:
            Perform a forward pass through the autoencoder.

    Inherits from:
        Autoencoder
    """

    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        loss_function: BaseBurgessLoss,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
        encoder_optimiser_params: dict[str, Any] | None = None,
        decoder_optimiser_params: dict[str, Any] | None = None,
    ):
        super().__init__()

        if encoder_optimiser_params is None:
            encoder_optimiser_params = {"lr": 1e-3}
        if decoder_optimiser_params is None:
            decoder_optimiser_params = {"lr": 1e-3, "weight_decay": 1e-7}

        self.ae_type = Autoencoders.BURGESS

        self.loss_function = loss_function
        self.observation_size = observation_size
        self.latent_dim = latent_dim

        self.encoder = BurgessEncoder(
            observation_size,
            latent_dim,
            num_layers,
            num_filters,
            kernel_size,
        )
        self.decoder = BurgessDecoder(
            observation_size,
            latent_dim,
            self.encoder.out_dim,
            num_layers,
            num_filters,
            kernel_size,
        )

        # Autoencoder Optimizer
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            **encoder_optimiser_params,
        )

        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            **decoder_optimiser_params,
        )

    def update_autoencoder(self, data: torch.Tensor) -> torch.Tensor:
        """
        Update the autoencoder parameters based on the given data.

        Args:
            data (torch.Tensor): The input data used for updating the autoencoder.

        Returns:
            torch.Tensor: The VAE loss after updating the autoencoder.
        """
        vae_loss = self.loss_function.update_autoencoder(data, self)
        return vae_loss

    def forward(
        self,
        observation,
        detach_cnn: bool = False,
        detach_output: bool = False,
    ) -> dict:
        """
        Perform a forward pass through the autoencoder.

        Args:
            observation: The input observation.
            detach_cnn (bool, optional): Whether to detach the CNN part of the encoder. Defaults to False.
            detach_output (bool, optional): Whether to detach the output of the encoder. Defaults to False.

        Returns:
            dict: A dictionary containing the latent observation, reconstructed observation, latent distribution, and loss.
        """
        latent_dist = self.encoder(
            observation, detach_cnn=detach_cnn, detach_output=detach_output
        )

        mu, logvar, latent_sample = latent_dist

        reconstructed_observation = self.decoder(latent_sample)

        # Train is false to get the full loss for the data
        loss = self.loss_function.calculate_loss(
            data=observation,
            reconstructed_data=reconstructed_observation,
            latent_dist=latent_dist,
            is_train=False,
        )

        if detach_output:
            latent_sample = latent_sample.detach()

        return {
            "latent_observation": latent_sample,
            "reconstructed_observation": reconstructed_observation,
            "latent_distribution": {"mu": mu, "logvar": logvar},
            "loss": loss,
        }


class BurgessEncoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        stride = 2
        padding = 0
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    observation_size[0],
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=padding,
                )
            ]
        )

        self.out_dim = hlp.flatten(
            observation_size[1], k=self.kernel_size, s=stride, p=padding  # type: ignore
        )

        stride = 1
        padding = 0
        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv2d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.out_dim = hlp.flatten(
                self.out_dim, k=self.kernel_size, s=stride, p=padding
            )

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        # Fully connected layers
        self.linear_one = nn.Linear(self.n_flatten, self.hidden_dim)
        self.linear_two = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(self.hidden_dim, self.latent_dim * 2)

    def copy_conv_weights_from(self, source):
        # Only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        conv = torch.relu(self.convs[0](x))

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def _enc_forward(
        self, obs: torch.Tensor, detach_cnn: bool = False, detach_output: bool = False
    ):
        batch_size = obs.size(0)

        # Convolutional layers with ReLu activations
        h = self._forward_conv(obs)

        # SAC AE detaches at the CNN layer
        if detach_cnn:
            h = h.detach()

        # Fully connected layers with ReLu activations
        h = h.view((batch_size, -1))
        h = torch.relu(self.linear_one(h))
        h = torch.relu(self.linear_two(h))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(h)

        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        if detach_output:
            mu = mu.detach()
            logvar = logvar.detach()

        latent_sample = self._reparameterize(mu, logvar)

        return mu, logvar, latent_sample

    def forward(self, obs, detach_cnn: bool = False, detach_output: bool = False):
        mu, logvar, latent_sample = self._enc_forward(
            obs, detach_cnn=detach_cnn, detach_output=detach_output
        )
        return mu, logvar, latent_sample


class BurgessDecoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        out_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        # Fully connected layers
        self.linear_one = nn.Linear(latent_dim, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, hidden_dim)
        self.linear_three = nn.Linear(hidden_dim, self.n_flatten)

        self.deconvs = nn.ModuleList()

        stride = 1
        padding = 0
        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_filters,
                    out_channels=self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

        stride = 2
        padding = 1
        self.deconvs.append(
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=observation_size[0],
                kernel_size=self.kernel_size,
                stride=stride,
                output_padding=padding,
            )
        )

    def forward(self, latent_obs):
        batch_size = latent_obs.size(0)

        # Fully connected layers with ReLu activations
        h_fc = torch.relu(self.linear_one(latent_obs))
        h_fc = torch.relu(self.linear_two(h_fc))
        h_fc = torch.relu(self.linear_three(h_fc))

        deconv = h_fc.view(batch_size, self.num_filters, self.out_dim, self.out_dim)

        # Convolutional layers with ReLu activations
        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        # Sigmoid activation for final conv layer
        observation = torch.sigmoid(self.deconvs[-1](deconv))

        return observation
