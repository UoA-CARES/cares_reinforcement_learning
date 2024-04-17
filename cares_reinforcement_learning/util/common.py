from typing import List

from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform
from torch.nn import functional as F


# Standard Multilayer Perceptron (MLP) network
class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
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
