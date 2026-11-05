import math
from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import RainbowIQNConfig
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear


class BaseRainbowIQNNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        feature_dim: int,
        quantile_embedding_dim: int,
        num_quantiles: int,
        num_action_quantiles: int,
        feature_layer: MLP | nn.Sequential,
        quantile_layer: MLP | nn.Sequential,
        value_stream: MLP | nn.Sequential,
        advantage_stream: MLP | nn.Sequential,
    ):
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
        )

        self.feature_dim = feature_dim
        self.quantile_embedding_dim = quantile_embedding_dim

        self.num_quantiles = num_quantiles
        self.num_action_quantiles = num_action_quantiles

        self.feature_layer = feature_layer
        self.quantile_layer = quantile_layer
        self.value_stream = value_stream
        self.advantage_stream = advantage_stream

        self.register_buffer(
            "cosine_basis",
            torch.arange(
                0,
                quantile_embedding_dim,
                dtype=torch.float32,
            ).view(1, 1, -1)
            * math.pi,
        )

    def forward(
        self,
        state: torch.Tensor,
        num_quantiles: int | None = None,
    ) -> torch.Tensor:
        quantiles, _ = self.quantiles(
            state,
            num_quantiles=num_quantiles or self.num_action_quantiles,
        )

        return quantiles.mean(dim=1)

    def quantiles(
        self,
        state: torch.Tensor,
        num_quantiles: int | None = None,
        taus: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        batch_size = state.shape[0]

        if taus is None:
            num_quantiles = (
                num_quantiles if num_quantiles is not None else self.num_quantiles
            )

            taus = torch.rand(
                batch_size,
                num_quantiles,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            if taus.ndim != 2 or taus.shape[0] != batch_size:
                raise ValueError("taus must have shape [batch_size, num_quantiles]")

            taus = taus.to(
                device=state.device,
                dtype=state.dtype,
            )

        num_quantiles = taus.shape[1]

        # State representation: [B, F]
        state_features = self.feature_layer(state)

        # Cosine embedding: [B, N, E]
        cosine_features = torch.cos(taus.unsqueeze(-1) * self.cosine_basis)

        # Quantile representation: [B, N, F]
        quantile_features = self.quantile_layer(cosine_features)

        # Combine state and quantile representations: [B, N, F]
        combined_features = state_features.unsqueeze(1) * quantile_features

        # Flatten quantile dimension for the dueling streams.
        # [B * N, F]
        combined_features = combined_features.reshape(
            batch_size * num_quantiles,
            self.feature_dim,
        )

        # Dueling value and advantage streams.
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)

        # [B, N, 1]
        value = value.view(
            batch_size,
            num_quantiles,
            1,
        )

        # [B, N, A]
        advantage = advantage.view(
            batch_size,
            num_quantiles,
            self.num_actions,
        )

        # Dueling decomposition for each quantile.
        quantile_values = value + advantage - advantage.mean(dim=2, keepdim=True)

        return quantile_values, taus

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def noise_stats(self) -> dict[str, Any]:
        stats = {}

        stats.update(self._module_noise_stats("feature_layer", self.feature_layer))

        stats.update(self._module_noise_stats("quantile_layer", self.quantile_layer))

        stats.update(self._module_noise_stats("value_stream", self.value_stream))

        stats.update(
            self._module_noise_stats("advantage_stream", self.advantage_stream)
        )

        return stats


# Default base network for testing
class DefaultNetwork(BaseRainbowIQNNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        feature_dim = 128
        quantile_embedding_dim = 64

        num_quantiles = 32
        num_action_quantiles = 32

        feature_layer = nn.Sequential(
            nn.Linear(observation_size, feature_dim),
            nn.ReLU(),
        )

        quantile_layer = nn.Sequential(
            nn.Linear(quantile_embedding_dim, feature_dim),
            nn.ReLU(),
        )

        value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, 1),
        )

        advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128),
            nn.ReLU(),
            NoisyLinear(128, num_actions),
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            feature_dim=feature_dim,
            quantile_embedding_dim=quantile_embedding_dim,
            num_quantiles=num_quantiles,
            num_action_quantiles=num_action_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )


class Network(BaseRainbowIQNNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: RainbowIQNConfig,
    ):
        feature_layer = MLP(
            input_size=observation_size,
            output_size=config.feature_dim,
            config=config.feature_layer_config,
        )

        quantile_layer = MLP(
            input_size=config.quantile_embedding_dim,
            output_size=config.feature_dim,
            config=config.quantile_layer_config,
        )

        value_stream = MLP(
            input_size=config.feature_dim,
            output_size=1,
            config=config.value_stream_config,
        )

        advantage_stream = MLP(
            input_size=config.feature_dim,
            output_size=num_actions,
            config=config.advantage_stream_config,
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            feature_dim=config.feature_dim,
            quantile_embedding_dim=config.quantile_embedding_dim,
            num_quantiles=config.num_quantiles,
            num_action_quantiles=config.num_action_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )
