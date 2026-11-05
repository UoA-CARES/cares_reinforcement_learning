"""
Fully Parameterized Quantile Function (FQF) network.

Paper: https://arxiv.org/abs/1911.02140

FQF learns both the quantile fractions and quantile values
used to approximate the return distribution.
"""

import math

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import FQFConfig
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseFQFNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        feature_dim: int,
        quantile_embedding_dim: int,
        num_quantiles: int,
        feature_layer: MLP | nn.Sequential,
        quantile_layer: MLP | nn.Sequential,
        value_layer: MLP | nn.Sequential,
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
        self.value_layer = value_layer

        # Fraction proposal network.
        self.fraction_layer = nn.Linear(
            feature_dim,
            num_quantiles,
        )

        # Initialise with uniform quantile intervals.
        nn.init.zeros_(self.fraction_layer.weight)
        nn.init.zeros_(self.fraction_layer.bias)

        # Cosine embedding frequencies: [0, 1, ..., E-1].
        self.register_buffer(
            "cosine_basis",
            torch.arange(
                0,
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

        # Fraction proposal probabilities: [B, N]
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

        # Midpoints of learned quantile intervals: [B, N]
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

        taus = taus.to(device=state_features.device, dtype=state_features.dtype)

        num_quantiles = taus.shape[1]

        # Cosine embedding: [B, N, E]
        cosine_features = torch.cos(taus.unsqueeze(-1) * self.cosine_basis)

        # Quantile representation: [B, N, F]
        quantile_features = self.quantile_layer(cosine_features)

        # Combined state and quantile features: [B, N, F]
        combined_features = state_features.unsqueeze(1) * quantile_features

        # Predict quantile values for each action.
        quantile_values = self.value_layer(
            combined_features.reshape(
                batch_size * num_quantiles,
                self.feature_dim,
            )
        )

        quantile_values = quantile_values.view(
            batch_size,
            num_quantiles,
            self.num_actions,
        )

        return quantile_values

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        # Calculate state features once.
        state_features = self.feature_layer(state)

        # Learn quantile fractions.
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

        # Expected action values weighted by quantile interval widths.
        q_values = (quantile_values * probabilities.unsqueeze(-1)).sum(dim=1)

        return q_values


# Default base network for testing
class DefaultNetwork(BaseFQFNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        feature_dim = 256
        quantile_embedding_dim = 64
        num_quantiles = 32

        feature_layer = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.ReLU(),
            nn.Linear(256, feature_dim),
            nn.ReLU(),
        )

        quantile_layer = nn.Sequential(
            nn.Linear(quantile_embedding_dim, feature_dim),
            nn.ReLU(),
        )

        value_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions),
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            feature_dim=feature_dim,
            quantile_embedding_dim=quantile_embedding_dim,
            num_quantiles=num_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_layer=value_layer,
        )


class Network(BaseFQFNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: FQFConfig,
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

        value_layer = MLP(
            input_size=config.feature_dim,
            output_size=num_actions,
            config=config.value_layer_config,
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            feature_dim=config.feature_dim,
            quantile_embedding_dim=config.quantile_embedding_dim,
            num_quantiles=config.num_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_layer=value_layer,
        )
