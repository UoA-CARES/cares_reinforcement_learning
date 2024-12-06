from typing import Any, Callable

import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform


def get_pytorch_module_from_name(module_name: str) -> Callable[..., nn.Module]:
    return getattr(nn, module_name)


# Standard Multilayer Perceptron (MLP) network
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_sizes: list[int],
        output_size: int | None,
        dropout_layer: Callable[..., nn.Module] | str | None = None,
        dropout_layer_args: dict[str, Any] | None = None,
        norm_layer: Callable[..., nn.Module] | str | None = None,
        norm_layer_args: dict[str, Any] | None = None,
        hidden_activation_function: Callable[..., nn.Module] | str = nn.ReLU,
        hidden_activation_function_args: dict[str, Any] | None = None,
        output_activation_function: Callable[..., nn.Module] | str | None = None,
        output_activation_args: dict[str, Any] | None = None,
    ):
        super().__init__()
        if dropout_layer_args is None:
            dropout_layer_args = {}
        if norm_layer_args is None:
            norm_layer_args = {}
        if hidden_activation_function_args is None:
            hidden_activation_function_args = {}
        if output_activation_args is None:
            output_activation_args = {}

        if isinstance(dropout_layer, str):
            dropout_layer = get_pytorch_module_from_name(dropout_layer)

        if isinstance(norm_layer, str):
            norm_layer = get_pytorch_module_from_name(norm_layer)

        if isinstance(hidden_activation_function, str):
            hidden_activation_function = get_pytorch_module_from_name(
                hidden_activation_function
            )

        if isinstance(output_activation_function, str):
            output_activation_function = get_pytorch_module_from_name(
                output_activation_function
            )

        layers = nn.ModuleList()

        for next_size in hidden_sizes:
            layers.append(nn.Linear(input_size, next_size))

            if dropout_layer is not None:
                layers.append(dropout_layer(**dropout_layer_args))

            if norm_layer is not None:
                layers.append(norm_layer(next_size, **norm_layer_args))

            layers.append(hidden_activation_function(**hidden_activation_function_args))

            input_size = next_size

        if output_size is not None:
            layers.append(nn.Linear(input_size, output_size))

            if output_activation_function is not None:
                layers.append(output_activation_function(**output_activation_args))

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
