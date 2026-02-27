from typing import Callable

import torch
from torch import nn

from cares_reinforcement_learning.networks.batchrenorm import BatchRenorm1d
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear
from cares_reinforcement_learning.algorithm.configurations import (
    FunctionLayer,
    MLPConfig,
    NormLayer,
    ResidualLayer,
    TrainableLayer,
)


def _get_pytorch_module_from_name(module_name: str) -> Callable[..., nn.Module]:
    if hasattr(nn, module_name):
        return getattr(nn, module_name)
    elif module_name == "BatchRenorm1d":
        return BatchRenorm1d
    elif module_name == "NoisyLinear":
        return NoisyLinear
    raise ValueError(f"Module {module_name} not found in nn or custom modules.")


# Standard Multilayer Perceptron (MLP) network - consider making Sequential itself
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None,
        config: MLPConfig,
    ):
        super().__init__()

        self.input_size = input_size

        layers = nn.ModuleList()

        current_input_size = self.input_size
        current_output_size = self.input_size

        for layer_spec in config.layers:
            if isinstance(layer_spec, TrainableLayer):
                if layer_spec.in_features is not None:
                    current_input_size = layer_spec.in_features

                if layer_spec.out_features is not None:
                    current_output_size = layer_spec.out_features
                elif output_size is not None:
                    current_output_size = output_size

                layer = _get_pytorch_module_from_name(layer_spec.layer_type)(
                    current_input_size, current_output_size, **layer_spec.params
                )
            elif isinstance(layer_spec, FunctionLayer):
                layer = _get_pytorch_module_from_name(layer_spec.layer_type)(
                    **layer_spec.params
                )
            elif isinstance(layer_spec, NormLayer):
                if layer_spec.in_features is not None:
                    current_input_size = layer_spec.in_features

                layer = _get_pytorch_module_from_name(layer_spec.layer_type)(
                    current_input_size, **layer_spec.params
                )
            elif isinstance(layer_spec, ResidualLayer):
                layer = MLPResidualBlock(current_input_size, layer_spec)
            else:
                raise ValueError(f"Unknown layer type {layer_spec}")

            layers.append(layer)

            current_input_size = current_output_size

        self.model = nn.Sequential(*layers)

        self.output_size = current_input_size if output_size is None else output_size

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        return self.model(input_value)


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        config: ResidualLayer,
    ):
        super().__init__()

        self.model = MLP(input_size, None, MLPConfig(layers=config.main_layers))

        self.input_size = input_size
        self.output_size = self.model.output_size

        self.use_padding = config.use_padding

        self.shortcut: nn.Module
        if config.shortcut_layer is None:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = MLP(
                input_size, None, MLPConfig(layers=[config.shortcut_layer])
            )

    def forward(self, input_value: torch.Tensor) -> torch.Tensor:
        main_output = self.model(input_value)
        if self.use_padding:
            input_value = self.pad_channels(input_value, self.output_size)
        return main_output + self.shortcut(input_value)

    def pad_channels(self, x: torch.Tensor, target_channels: int):
        """
        Pads tensor `x` along channel dimension (dim=1) with zeros
        until it reaches `target_channels`.
        Works for shapes [N, C, L], [N, C, H, W], or [N, C, H, W, T].
        """
        N, C = x.shape[:2]
        if C >= target_channels:
            return x

        pad_channels = target_channels - C
        # Create a zero tensor of the same type and device
        pad_shape = (N, pad_channels, *x.shape[2:])
        zeros = torch.zeros(pad_shape, dtype=x.dtype, device=x.device)
        return torch.cat([x, zeros], dim=1)
