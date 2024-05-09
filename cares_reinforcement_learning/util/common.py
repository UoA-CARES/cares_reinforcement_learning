import numpy as np
import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F


# Standard Multilayer Perceptron (MLP) network
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: list[int], output_size: int):
        super().__init__()

        self.fully_connected_layers = []
        for i, next_size in enumerate(hidden_sizes):
            fully_connected_layer = nn.Linear(input_size, next_size)
            self.add_module(f"fully_connected_layer_{i}", fully_connected_layer)
            self.fully_connected_layers.append(fully_connected_layer)
            input_size = next_size

        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, state):
        for fully_connected_layer in self.fully_connected_layers:
            state = F.relu(fully_connected_layer(state))
        output = self.output_layer(state)
        return output


# CNN from Nature paper: https://www.nature.com/articles/nature14236
class NatureCNN(nn.Module):
    def __init__(self, observation_size: tuple[int]):
        super().__init__()

        self.cnn_modules = [
            nn.Conv2d(observation_size[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        ]

        self.nature_cnn = nn.Sequential(*self.cnn_modules)

        with torch.no_grad():
            dummy_image = torch.zeros([1, *observation_size])
            n_flatten = self.nature_cnn(torch.FloatTensor(dummy_image))

        self.cnn_modules.append(nn.Linear(n_flatten.shape[1], 512))

        self.nature_cnn = nn.Sequential(*self.cnn_modules)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.nature_cnn(state)
        return output


class Encoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_filters = num_filters

        self.cov_net = nn.ModuleList(
            [nn.Conv2d(observation_size[0], self.num_filters, kernel_size=3, stride=2)]
        )
        for _ in range(self.num_layers - 1):
            self.cov_net.append(
                nn.Conv2d(self.num_filters, self.num_filters, kernel_size=3, stride=1)
            )

        with torch.no_grad():
            dummy_image = torch.zeros([1, *observation_size])
            n_flatten = self._forward_conv(torch.FloatTensor(dummy_image)).shape[1]

        self.fc = nn.Linear(n_flatten, self.latent_dim)
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
        out = torch.tanh(h_norm)
        if detach:
            out = out.detach()
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        n_flatten: int,
        num_layers: int = 4,
        num_filters: int = 32,
    ):
        super().__init__()

        self.latent_dim = latent_dim

        self.num_layers = num_layers
        self.num_filters = num_filters

        # resolve 39200
        self.fc_1 = nn.Linear(self.latent_dim, n_flatten)

        self.deconvs = nn.ModuleList()

        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_filters,
                    out_channels=self.num_filters,
                    kernel_size=3,
                    stride=1,
                )
            )

        self.deconvs.append(
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=observation_size[0],
                kernel_size=3,
                stride=2,
                output_padding=1,
            )
        )
        # nn.Sequential(
        #     nn.ConvTranspose2d(
        #         in_channels=self.num_filters,
        #         out_channels=self.num_filters,
        #         kernel_size=3,
        #         stride=1,
        #     ),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         in_channels=self.num_filters,
        #         out_channels=self.num_filters,
        #         kernel_size=3,
        #         stride=1,
        #     ),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         in_channels=self.num_filters,
        #         out_channels=self.num_filters,
        #         kernel_size=3,
        #         stride=1,
        #     ),
        #     nn.ReLU(),
        #     nn.ConvTranspose2d(
        #         in_channels=self.num_filters,
        #         out_channels=observation_size[0],
        #         kernel_size=3,
        #         stride=2,
        #         output_padding=1,
        #     ),
        #     nn.Sigmoid(),
        # )

        self.apply(weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc_1(x))
        x = x.view(-1, 32, 35, 35)
        x = self.deconvs(x)
        return x


# Stable version of the Tanh transform - overriden to avoid NaN values through atanh in pytorch
class StableTanhTransform(TanhTransform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)


# These methods are not required for the purposes of SAC and are thus intentionally ignored
# pylint: disable=abstract-method
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
