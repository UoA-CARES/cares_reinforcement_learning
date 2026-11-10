import math
from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import RainbowFQFConfig
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear


class BaseRainbowFQFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        feature_dim: int,
        quantile_embedding_dim: int,
        num_quantiles: int,
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

        self.feature_layer = feature_layer
        self.quantile_layer = quantile_layer
        self.value_stream = value_stream
        self.advantage_stream = advantage_stream

        # FQF fraction proposal network.
        # This is optimised separately from the value network.
        self.fraction_layer = nn.Linear(
            feature_dim,
            num_quantiles,
        )

        # Initialise with uniform quantile intervals.
        nn.init.zeros_(self.fraction_layer.weight)
        nn.init.zeros_(self.fraction_layer.bias)

        # Cosine embedding frequencies.
        self.cosine_basis: torch.Tensor
        self.register_buffer(
            "cosine_basis",
            torch.arange(
                quantile_embedding_dim,
                dtype=torch.float32,
            ).view(1, 1, -1)
            * math.pi,
        )

    def propose_fractions(
        self,
        state: torch.Tensor,
        state_features: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if state_features is None:
            state_features = self.feature_layer(state)

        # Learned interval probabilities: [B, N]
        logits = self.fraction_layer(state_features)
        probabilities = torch.softmax(logits, dim=-1)

        # Quantile boundaries: [B, N+1]
        fractions = torch.cat(
            (
                torch.zeros_like(probabilities[:, :1]),
                probabilities.cumsum(dim=1),
            ),
            dim=1,
        )

        # Interval midpoints: [B, N]
        midpoints = (fractions[:, :-1] + fractions[:, 1:]) / 2.0

        return fractions, midpoints, probabilities

    def quantiles(
        self,
        state: torch.Tensor,
        taus: torch.Tensor,
        state_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = state.shape[0]

        if taus.ndim != 2 or taus.shape[0] != batch_size:
            raise ValueError("taus must have shape [batch_size, num_quantiles]")

        if state_features is None:
            state_features = self.feature_layer(state)

        taus = taus.to(
            device=state_features.device,
            dtype=state_features.dtype,
        )

        num_quantiles = taus.shape[1]

        # Cosine embedding: [B, N, E]
        cosine_features = torch.cos(taus.unsqueeze(-1) * self.cosine_basis)

        # Quantile representation: [B, N, F]
        quantile_features = self.quantile_layer(cosine_features)

        # Combined representation: [B, N, F]
        combined_features = state_features.unsqueeze(1) * quantile_features

        # Flatten quantile dimension for the dueling streams.
        combined_features = combined_features.reshape(
            batch_size * num_quantiles,
            self.feature_dim,
        )

        # Dueling value and advantage streams.
        value = self.value_stream(combined_features)
        advantage = self.advantage_stream(combined_features)

        value = value.view(
            batch_size,
            num_quantiles,
            1,
        )

        advantage = advantage.view(
            batch_size,
            num_quantiles,
            self.num_actions,
        )

        # Normalise advantage across actions, per quantile.
        quantile_values = value + advantage - advantage.mean(dim=2, keepdim=True)

        return quantile_values

    def forward(
        self,
        state: torch.Tensor,
    ) -> torch.Tensor:
        # Calculate state representation once.
        state_features = self.feature_layer(state)

        # Propose learned quantile intervals.
        _, midpoints, probabilities = self.propose_fractions(
            state,
            state_features=state_features,
        )

        # Predict quantile values at the learned midpoints.
        quantile_values = self.quantiles(
            state,
            taus=midpoints,
            state_features=state_features,
        )

        # Expected Q-values weighted by interval probabilities.
        q_values = (quantile_values * probabilities.unsqueeze(-1)).sum(dim=1)

        return q_values

    def reset_noise(self) -> None:
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def noise_stats(self) -> dict[str, Any]:
        stats: dict[str, Any] = {}

        for name in (
            "feature_layer",
            "quantile_layer",
            "fraction_layer",
            "value_stream",
            "advantage_stream",
        ):
            stats.update(
                self._module_noise_stats(
                    name,
                    getattr(self, name),
                )
            )

        return stats


# Default base network for testing
class DefaultNetwork(BaseRainbowFQFNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        feature_dim = 128
        quantile_embedding_dim = 64
        num_quantiles = 32

        feature_layer = nn.Sequential(
            nn.Linear(observation_size, feature_dim),
            nn.ReLU(),
        )

        quantile_layer = nn.Sequential(
            nn.Linear(quantile_embedding_dim, feature_dim),
            nn.ReLU(),
        )

        value_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128, sigma_init=1.0),
            nn.ReLU(),
            NoisyLinear(128, 1, sigma_init=0.5),
        )

        advantage_stream = nn.Sequential(
            NoisyLinear(feature_dim, 128, sigma_init=1.0),
            nn.ReLU(),
            NoisyLinear(128, num_actions, sigma_init=0.5),
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            feature_dim=feature_dim,
            quantile_embedding_dim=quantile_embedding_dim,
            num_quantiles=num_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )


class Network(BaseRainbowFQFNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: RainbowFQFConfig,
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
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )
