import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.encoders.autoencoder import Autoencoder
from cares_reinforcement_learning.networks.encoders.losses import AELoss


def tie_weights(src, trg):
    trg.weight = src.weight
    trg.bias = src.bias


class VanillaAutoencoder(Autoencoder):
    """
    An image based autoencoder model consisting of an encoder and a decoder pair.

    Args:
        observation_size (tuple[int]): The size of the input image observations.
        latent_dim (int): The dimension of the latent space.
        num_layers (int, optional): The number of layers in the encoder and decoder. Defaults to 4.
        num_filters (int, optional): The number of filters in each layer. Defaults to 32.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.
    """

    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__(
            loss_function=AELoss(),
            observation_size=observation_size,
            latent_dim=latent_dim,
        )

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

    def forward(
        self,
        observation: torch.Tensor,
        detach_cnn: bool = False,
        detach_output: bool = False,
        **kwargs,
    ) -> torch.Tensor:
        latent_observation = self.encoder(observation, detach_cnn, detach_output)

        reconstructed_observation = self.decoder(latent_observation)

        loss = self.loss_function(
            data=observation,
            reconstructed_data=reconstructed_observation,
            latent_sample=latent_observation,
        )

        return {"reconstructed_observation": reconstructed_observation, "loss": loss}


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

        self.out_dim = hlp.flatten(observation_size[1], k=self.kernel_size, s=2)

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
        latent_obs = torch.tanh(h_norm)

        # NaSATD3 detatches the encoder output
        if detach_output:
            latent_obs = latent_obs.detach()

        return latent_obs


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

    def forward(self, latent_obs: torch.Tensor) -> torch.Tensor:
        h_fc = self.fc(latent_obs)
        h_fc = torch.relu(h_fc)

        deconv = h_fc.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        observation = self.deconvs[-1](deconv)

        return observation
