import math
from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp import MLP
from cares_reinforcement_learning.util.network_configurations import MLPConfig
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
        custom_encoder_config: MLPConfig | None = None,
        detach_at_convs: bool = True,
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

        self.encoder = NewEncoder(
            self.observation_size,
            self.latent_dim,
            num_layers,
            num_filters,
            kernel_size,
            custom_network_config=custom_encoder_config,
            detach_at_convs=detach_at_convs,
        )

        encoder_network = None
        if custom_encoder_config is not None:
            encoder_network = self.encoder.convs

        self.decoder = NewDecoder(
            self.observation_size,
            self.latent_dim,
            self.encoder.out_height,
            num_layers,
            num_filters,
            kernel_size,
            encoder_network=encoder_network
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

        self.out_height = hlp.flatten(observation_size[1], k=self.kernel_size, s=2)  # type: ignore

        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv2d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )
            self.convs.append(nn.ReLU())
            self.out_height = hlp.flatten(self.out_height, k=self.kernel_size, s=1)

        self.n_flatten = self.out_height * self.out_height * self.num_filters

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


class Detach(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.detach()


class NewEncoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
        custom_network_config: MLPConfig | None = None,
        detach_at_convs: bool = True,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        if custom_network_config is None:
            self.num_layers = num_layers
            self.num_filters = num_filters
            self.kernel_size = kernel_size

            self.convs = nn.Sequential(
                [
                    nn.Conv2d(
                        observation_size[0],
                        self.num_filters,
                        kernel_size=self.kernel_size,
                        stride=2,
                    ),
                    nn.ReLU(),
                ]
            )

            self.out_height = hlp.flatten(observation_size[1], k=self.kernel_size, s=2)

            for _ in range(self.num_layers - 1):
                self.convs.append(
                    nn.Conv2d(
                        self.num_filters,
                        self.num_filters,
                        kernel_size=self.kernel_size,
                        stride=1,
                    )
                )
                self.convs.append(nn.ReLU())

            self.out_height = hlp.flatten(self.out_height, k=self.kernel_size, s=1)
        else:
            self.convs = MLP(
                input_size=observation_size,
                output_size=None,
                config=custom_network_config
            )
            self.out_height = self.convs.conv_output_shape[1]  # for backwards compatibility with old Encoder class

        self.n_flatten = math.prod(self.convs.conv_output_shape)
        
        self.detach_at_convs = detach_at_convs
        self.encoders = []

        fc_layer = self.create_fc_layer()
        self.encoder_no_detach = nn.Sequential(
            self.convs,
            fc_layer,
        ).to(hlp.get_device())

        # Instantiate default encoder based on detach behaviour
        if not self.detach_at_convs:
            self.encoders.append(nn.Sequential(
                self.encoder_no_detach,
                Detach(),
            ).to(hlp.get_device()))
            # If detach is at end all encoders are the same
        else:
            self.encoders.append(nn.Sequential(
                self.convs,
                Detach(),
                fc_layer,
            ).to(hlp.get_device()))


    def create_fc_layer(self) -> nn.Sequential:
        fc_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.n_flatten, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.Tanh(),
        )
        return fc_layer
    
    
    def get_detached_encoder(self) -> nn.Sequential:
        """Retrieves an encoder network based on the detach configuration of the autoencoder. 

        Only direct calls to the encoder `encoder(obs)` or more explicitly `encoder.encoder_no_detach(obs)` 
        preserve full gradient flow for AE training.
        """
        if not self.detach_at_convs:
            return self.encoders[0]
        
        fc_layer = self.create_fc_layer()
        encoder = nn.Sequential(
            self.convs,
            Detach(),
            fc_layer,
        ).to(hlp.get_device())

        self.encoders.append(encoder)
        return encoder
    

    def get_encoder(self) -> nn.Sequential:
        """Gets the encoder network based on the detach configuration of the autoencoder."""
        return self.encoder_no_detach


    def forward(self, obs: torch.Tensor, detach_cnn: bool = False, detach_output: bool = False) -> torch.Tensor:
        """Forward pass of the encoder without detach"""
        return self.encoder_no_detach(obs)
    

class Decoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        out_height: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_height

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


class NewDecoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        out_height: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
        encoder_network: MLP | None = None,
    ):
        super().__init__()
        self.observation_size = observation_size

        self.latent_dim = latent_dim
        self.out_height = out_height
        self.out_width = encoder_network.conv_output_shape[2] if encoder_network is not None else out_height
        out_channels = encoder_network.conv_output_shape[0] if encoder_network is not None else num_filters

        # Map from latent to output dim of encoder network when flattened
        self.n_flatten = self.out_height * self.out_width * out_channels

        self.deconvs = nn.Sequential(
            nn.Linear(latent_dim, self.n_flatten),
            nn.ReLU(),
            nn.Unflatten(1, (out_channels, self.out_height, self.out_width)),
        )

        if encoder_network is None:
            self.num_layers = num_layers
            self.num_filters = num_filters
            self.kernel_size = kernel_size

            for _ in range(self.num_layers - 1):
                self.deconvs.append(
                    nn.ConvTranspose2d(
                        in_channels=self.num_filters,
                        out_channels=self.num_filters,
                        kernel_size=self.kernel_size,
                        stride=1,
                    )
                )
                self.deconvs.append(nn.ReLU())

            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_filters,
                    out_channels=observation_size[0],
                    kernel_size=self.kernel_size,
                    stride=2,
                    output_padding=1,
                )
            )
        else:
            self.invert_encoder(encoder_network.model)
        
        self.deconvs.append(nn.Sigmoid())
        self.deconvs.to(hlp.get_device())


    def invert_encoder(self, encoder_network: nn.Module) -> None:
        for i in range(len(encoder_network) - 1, 0, -1): # Assumes first layer is conv
            layer = encoder_network[i]
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Flatten):
                continue
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=layer.out_channels,
                    out_channels=layer.in_channels,
                    kernel_size=layer.kernel_size,
                    stride=layer.stride,
                    padding=layer.padding,
                    output_padding=layer.stride[0] - 1
                )
            )
            self.deconvs.append(nn.ReLU())
        
        self.deconvs.append(nn.ConvTranspose2d(
            in_channels=encoder_network[0].out_channels,
            out_channels=encoder_network[0].in_channels,
            kernel_size=encoder_network[0].kernel_size,
            stride=encoder_network[0].stride,
            padding=encoder_network[0].padding,
            output_padding=encoder_network[0].stride[0] - 1
        ))


    def forward(self, latent_observation: torch.Tensor) -> torch.Tensor:
        output = self.deconvs(latent_observation)
        return nn.functional.interpolate(output, size=(self.observation_size[1], self.observation_size[2]))