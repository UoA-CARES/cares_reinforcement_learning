from typing import Any

import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.losses import AELoss


def tie_weights(src, trg):
    trg.weight = src.weight
    trg.bias = src.bias


class VanillaAutoencoder(nn.Module):
    """
    An image-based autoencoder model consisting of an encoder and a decoder pair.

    Args:
        observation_size (tuple[int]): The size of the input image observations.
        latent_dim (int): The dimension of the latent space.
        num_layers (int, optional): The number of layers in the encoder and decoder. Defaults to 4.
        num_filters (int, optional): The number of filters in each layer. Defaults to 32.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
        latent_lambda (float, optional): The weight of the latent regularization term in the loss function. Defaults to 1e-6.
        encoder_optimiser_params (dict[str, any], optional): Additional parameters for the encoder optimizer. Defaults to {"lr": 1e-4}.
        decoder_optimiser_params (dict[str, any], optional): Additional parameters for the decoder optimizer. Defaults to {"lr": 1e-4}.

    Attributes:
        encoder (Encoder): The encoder component of the autoencoder.
        decoder (Decoder): The decoder component of the autoencoder.
        encoder_optimizer (torch.optim.Adam): The optimizer for the encoder.
        decoder_optimizer (torch.optim.Adam): The optimizer for the decoder.

    Methods:
        update_autoencoder(data: torch.Tensor) -> None:
            Update the autoencoder model by performing a forward pass and backpropagation.

        forward(observation: torch.Tensor, detach_cnn: bool = False, detach_output: bool = False, **kwargs) -> torch.Tensor:
            Perform a forward pass through the autoencoder model.

    Inherits from:
        Autoencoder
    """

    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
        latent_lambda: float = 1e-6,
        encoder_optimiser_params: dict[str, Any] | None = None,
        decoder_optimiser_params: dict[str, Any] | None = None,
    ):
        super().__init__()

        if encoder_optimiser_params is None:
            encoder_optimiser_params = {"lr": 1e-4}
        if decoder_optimiser_params is None:
            decoder_optimiser_params = {"lr": 1e-4}

        self.ae_type = Autoencoders.AE

        self.loss_function = AELoss(latent_lambda=latent_lambda)

        self.observation_size = observation_size
        self.latent_dim = latent_dim

        self.encoder = Encoder(
            self.observation_size,
            self.latent_dim,
            num_layers,
            num_filters,
            kernel_size,
        )

        self.decoder = Decoder(
            self.observation_size,
            self.latent_dim,
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
            float: The AE loss after updating the autoencoder.
        """

        ae_loss = self.loss_function.update_autoencoder(data, self)
        return ae_loss

    def forward(
        self,
        observation: torch.Tensor,
        detach_cnn: bool = False,
        detach_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Perform a forward pass through the autoencoder model.

        Args:
            observation (torch.Tensor): The input observation to be encoded and decoded.
            detach_cnn (bool, optional): Whether to detach the CNN part of the encoder. Defaults to False.
            detach_output (bool, optional): Whether to detach the output of the encoder. Defaults to False.
            **kwargs: Additional keyword arguments.

        Returns:
            torch.Tensor: The reconstructed observation and the latent observation.
        """
        latent_observation = self.encoder(
            observation, detach_cnn=detach_cnn, detach_output=detach_output
        )

        reconstructed_observation = self.decoder(latent_observation)

        loss = self.loss_function.calculate_loss(
            data=observation,
            reconstructed_data=reconstructed_observation,
            latent_sample=latent_observation,
        )

        return {
            "latent_observation": latent_observation,
            "reconstructed_observation": reconstructed_observation,
            "loss": loss,
        }


class Encoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    observation_size[0],
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=2,
                )
            ]
        )

        self.out_dim = hlp.flatten(observation_size[1], k=self.kernel_size, s=2)  # type: ignore

        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv2d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )
            self.out_dim = hlp.flatten(self.out_dim, k=self.kernel_size, s=1)

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        self.fc = nn.Linear(self.n_flatten, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)

    def copy_conv_weights_from(self, source):
        # Only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        conv = torch.relu(self.convs[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        h = torch.flatten(conv, start_dim=1)
        return h

    def forward(
        self, obs: torch.Tensor, detach_cnn: bool = False, detach_output: bool = False
    ) -> torch.Tensor:
        h = self._forward_conv(obs)

        # SAC AE detaches at the CNN layer
        if detach_cnn:
            h = h.detach()

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        latent_observation = torch.tanh(h_norm)

        # NaSATD3 detatches the encoder output
        if detach_output:
            latent_observation = latent_observation.detach()

        return latent_observation


class Decoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        out_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        self.fc = nn.Linear(self.latent_dim, self.n_flatten)

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_filters,
                    out_channels=self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )

        self.deconvs.append(
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=observation_size[0],
                kernel_size=self.kernel_size,
                stride=2,
                output_padding=1,
            )
        )

    def forward(self, latent_observation: torch.Tensor) -> torch.Tensor:
        h_fc = self.fc(latent_observation)
        h_fc = torch.relu(h_fc)

        deconv = h_fc.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        observation = torch.sigmoid(self.deconvs[-1](deconv))

        return observation
