"""Dueling NoisyNet quantile network for RainbowQR."""

from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import RainbowQRConfig
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear


class BaseRainbowQRNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        quantiles: int,
        feature_layer: nn.Module,
        value_stream: nn.Module,
        advantage_stream: nn.Module,
    ):
        super().__init__(observation_size=observation_size, num_actions=num_actions)

        self.quantiles = quantiles
        self.feature_layer = feature_layer
        self.value_stream = value_stream
        self.advantage_stream = advantage_stream

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.calculate_quantiles(state).mean(dim=-1)

    def calculate_quantiles(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)

        # Dueling decomposition is performed separately at each quantile.
        value = self.value_stream(features).view(-1, 1, self.quantiles)
        advantage = self.advantage_stream(features).view(
            -1, self.num_actions, self.quantiles
        )

        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self) -> None:
        for stream in (self.feature_layer, self.value_stream, self.advantage_stream):
            for module in stream.modules():
                if isinstance(module, NoisyLinear):
                    module.reset_noise()

    def noise_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}
        stats.update(self._module_noise_stats("feature_layer", self.feature_layer))
        stats.update(self._module_noise_stats("value_stream", self.value_stream))
        stats.update(
            self._module_noise_stats("advantage_stream", self.advantage_stream)
        )
        return stats


class DefaultNetwork(BaseRainbowQRNetwork):
    def __init__(self, observation_size: int, num_actions: int):
        quantiles = 200
        hidden_size = 128

        feature_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.ReLU(),
        )

        value_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, quantiles),
        )

        advantage_stream = nn.Sequential(
            NoisyLinear(hidden_size, hidden_size),
            nn.ReLU(),
            NoisyLinear(hidden_size, num_actions * quantiles),
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            quantiles=quantiles,
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )


class Network(BaseRainbowQRNetwork):
    def __init__(
        self, observation_size: int, num_actions: int, config: RainbowQRConfig
    ):
        feature_layer = MLP(
            input_size=observation_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        value_stream = MLP(
            input_size=feature_layer.output_size,
            output_size=config.quantiles,
            config=config.value_stream_config,
        )

        advantage_stream = MLP(
            input_size=feature_layer.output_size,
            output_size=num_actions * config.quantiles,
            config=config.advantage_stream_config,
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            quantiles=config.quantiles,
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )
