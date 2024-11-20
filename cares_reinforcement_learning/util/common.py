from typing import Callable

import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F


# Standard Multilayer Perceptron (MLP) network
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int,
        norm_layer: Callable[..., nn.Module] | None = None,
        activation_function: Callable[..., torch.nn.Module] = nn.ReLU,
    ):
        super().__init__()

        layers = nn.ModuleList()

        for next_size in hidden_sizes:
            layers.append(nn.Linear(input_size, next_size))
            if norm_layer is not None:
                layers.append(norm_layer())
            layers.append(activation_function())
            input_size = next_size

        layers.append(nn.Linear(input_size, output_size))

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)


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
