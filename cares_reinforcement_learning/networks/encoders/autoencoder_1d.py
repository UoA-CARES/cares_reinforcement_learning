import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp


def tie_weights(src, trg):
    trg.weight = src.weight
    trg.bias = src.bias


class Encoder1D(nn.Module):
    def __init__(
        self,
        observation_size: int,
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

        # initialize first convolution layer: 1 channel in, stride 2
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    1,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=2,
                )
            ]
        )
        # size of each channel after convolution
        self.out_dim = hlp.flatten(observation_size, k=self.kernel_size, s=2)

        # Add rest of the layers
        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv1d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )
            # calculate current size of each channel
            self.out_dim = hlp.flatten(self.out_dim, k=self.kernel_size, s=1)

        self.n_flatten = self.out_dim * self.num_filters

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


class Decoder1D(nn.Module):
    def __init__(
        self,
        observation_size: int,
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

        self.n_flatten = self.out_dim * self.num_filters

        self.fc = nn.Linear(self.latent_dim, self.n_flatten)

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose1d(
                    in_channels=self.num_filters,
                    out_channels=self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )

        self.deconvs.append(
            nn.ConvTranspose1d(
                in_channels=self.num_filters,
                out_channels=1,
                kernel_size=self.kernel_size,
                stride=2,
                output_padding=1,
            )
        )

    def forward(self, latent_obs: torch.Tensor) -> torch.Tensor:
        h_fc = self.fc(latent_obs)
        h_fc = torch.relu(h_fc)

        deconv = h_fc.view(-1, self.num_filters, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        observation = self.deconvs[-1](deconv)

        return observation


def create_autoencoder_1d(
    observation_size: int,
    latent_dim: int,
    num_layers: int = 4,
    num_filters: int = 32,
    kernel_size: int = 3,
) -> tuple[nn.Module, nn.Module]:
    """
    Creates an autoencoder model consisting of an encoder and a decoder pair for 1D data.

    Args:
        observation_size (int): The size of the input image observations.
        latent_dim (int): The dimension of the latent space.
        num_layers (int, optional): The number of layers in the encoder and decoder. Defaults to 4.
        num_filters (int, optional): The number of filters in each layer. Defaults to 32.
        kernel_size (int, optional): The size of the convolutional kernel. Defaults to 3.

    Returns:
        tuple[nn.Module, nn.Module]: A tuple containing the encoder and decoder modules.
    """
    encoder = Encoder1D(
        observation_size, latent_dim, num_layers, num_filters, kernel_size
    )

    decoder = Decoder1D(
        observation_size,
        latent_dim,
        encoder.out_dim,
        num_layers,
        num_filters,
        kernel_size,
    )
    return encoder, decoder
