import math

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import IQNConfig
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseIQNNetwork(BaseNetwork):
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
        value_layer: MLP | nn.Sequential,
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
        self.value_layer = value_layer

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
            state, num_quantiles=num_quantiles or self.num_action_quantiles
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

        # Predict action values for each sampled quantile: [B, N, A]
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

        return quantile_values, taus


# Default base network for testing
class DefaultNetwork(BaseIQNNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        feature_dim = 256
        quantile_embedding_dim = 64

        num_quantiles = 32
        num_action_quantiles = 32

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
            num_action_quantiles=num_action_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_layer=value_layer,
        )


class Network(BaseIQNNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: IQNConfig,
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
            num_action_quantiles=config.num_action_quantiles,
            feature_layer=feature_layer,
            quantile_layer=quantile_layer,
            value_layer=value_layer,
        )
