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
        norm_layer_parameters: (
            tuple[Callable[..., nn.Module], dict[str, Any]]
            | tuple[str, dict[str, Any]]
            | None
        ) = None,
        activation_function_parameters: (
            tuple[Callable[..., nn.Module], dict[str, Any]] | tuple[str, dict[str, Any]]
        ) = (nn.ReLU, {}),
        final_activation_parameters: (
            tuple[Callable[..., nn.Module], dict[str, Any]]
            | tuple[str, dict[str, Any]]
            | None
        ) = None,
    ):
        super().__init__()

        norm_layer: Callable[..., nn.Module] | None = None
        norm_layer_args: dict[str, Any] = {}

        if norm_layer_parameters is not None:
            if isinstance(norm_layer_parameters[0], str):
                norm_layer = get_pytorch_module_from_name(norm_layer_parameters[0])
            else:
                norm_layer = norm_layer_parameters[0]
            norm_layer_args = norm_layer_parameters[1]

        if isinstance(activation_function_parameters[0], str):
            activation_function = get_pytorch_module_from_name(
                activation_function_parameters[0]
            )
        else:
            activation_function = activation_function_parameters[0]

        activation_function_args = activation_function_parameters[1]

        final_activation: Callable[..., nn.Module] | None = None
        final_activation_args: dict[str, Any] = {}

        if final_activation_parameters is not None:
            if isinstance(final_activation_parameters[0], str):
                final_activation = get_pytorch_module_from_name(
                    final_activation_parameters[0]
                )
            else:
                final_activation = final_activation_parameters[0]
            final_activation_args = final_activation_parameters[1]

        layers = nn.ModuleList()

        for next_size in hidden_sizes:
            layers.append(nn.Linear(input_size, next_size))

            if norm_layer is not None:
                layers.append(norm_layer(next_size, **norm_layer_args))

            layers.append(activation_function(**activation_function_args))

            input_size = next_size

        if output_size is not None:
            layers.append(nn.Linear(input_size, output_size))

            if final_activation is not None:
                layers.append(final_activation(**final_activation_args))

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
