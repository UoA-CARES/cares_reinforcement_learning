import numpy as np
import torch
from torch import nn


def flatten(w, k=3, s=1, p=0, m=True):
    """
    Returns the right size of the flattened tensor after convolutional transformation
    :param w: width of image
    :param k: kernel size
    :param s: stride
    :param p: padding
    :param m: max pooling (bool)
    :return: proper shape and params: use x * x * previous_out_channels

    Example:
    r = flatten(*flatten(*flatten(w=100, k=3, s=1, p=0, m=True)))[0]
    self.fc1 = nn.Linear(r*r*128, 1024)
    """
    return int((np.floor((w - k + 2 * p) / s) + 1) if m else 1)


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


def create_autoencoder(
    observation_size: tuple[int],
    latent_dim: int,
    num_layers: int = 4,
    num_filters: int = 32,
    kernel_size: int = 3,
) -> tuple[nn.Module, nn.Module]:

    encoder = Encoder(
        observation_size, latent_dim, num_layers, num_filters, kernel_size
    )

    decoder = Decoder(
        observation_size,
        latent_dim,
        encoder.out_dim,
        num_layers,
        num_filters,
        kernel_size,
    )
    return encoder, decoder


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

        self.cov_net = nn.ModuleList(
            [
                nn.Conv2d(
                    observation_size[0],
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=2,
                )
            ]
        )

        self.out_dim = flatten(observation_size[1], k=self.kernel_size, s=2)

        for _ in range(self.num_layers - 1):
            self.cov_net.append(
                nn.Conv2d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=1,
                )
            )
            self.out_dim = flatten(self.out_dim, k=self.kernel_size, s=1)

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        self.fc = nn.Linear(self.n_flatten, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)

        self.apply(weight_init)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        conv = torch.relu(self.cov_net[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.cov_net[i](conv))
        h = torch.flatten(conv, start_dim=1)
        return h

    def forward(self, obs: torch.Tensor, detach: bool = False) -> torch.Tensor:
        h = self._forward_conv(obs)

        h_fc = self.fc(h)
        h_norm = self.ln(h_fc)
        latent_obs = torch.tanh(h_norm)

        if detach:
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

        self.apply(weight_init)

    def forward(self, latent_obs: torch.Tensor) -> torch.Tensor:
        h_fc = self.fc(latent_obs)
        h_fc = torch.relu(h_fc)

        deconv = h_fc.view(-1, self.num_filters, self.out_dim, self.out_dim)

        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        observation = self.deconvs[-1](deconv)

        return observation
