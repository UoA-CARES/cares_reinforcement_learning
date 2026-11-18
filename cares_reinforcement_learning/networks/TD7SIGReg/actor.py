"""
TD7SIGReg actor - TD7's actor with the policy MLP split into an explicit
body (ending in a Linear layer) and head (activation + output layer).

This exposes two representations for the optional SIGReg regulariser:
    feature     - feature-layer output (the tensor fed to AvgL1Norm, pre-norm)
    penultimate - body output: pre-activation output of the last hidden Linear

AvgL1Norm on the feature layer is optional (config.actor_use_avg_l1_norm).
"""

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import TD7SIGRegConfig
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseActor(nn.Module):
    def __init__(
        self, observation_size: int, num_actions: int, config: TD7SIGRegConfig
    ):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.use_avg_l1_norm = bool(config.actor_use_avg_l1_norm)

        self.feature_layer: MLP | nn.Sequential = MLP(
            input_size=observation_size,
            output_size=config.zs_dim,
            config=config.feature_layer_config,
        )

        self.act_body: MLP | nn.Sequential = MLP(
            input_size=config.zs_dim + self.feature_layer.output_size,
            output_size=None,
            config=config.actor_body_config,
        )

        self.act_head: MLP | nn.Sequential = MLP(
            input_size=self.act_body.output_size,
            output_size=num_actions,
            config=config.actor_head_config,
        )

    def forward_with_features(
        self, state: torch.Tensor, zs: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature = self.feature_layer(state)

        latent = fnc.avg_l1_norm(feature) if self.use_avg_l1_norm else feature

        combined = torch.cat([latent, zs], dim=-1)

        penultimate = self.act_body(combined)
        output = self.act_head(penultimate)

        return output, {"feature": feature, "penultimate": penultimate}

    def forward(self, state: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_features(state, zs)
        return output


class DefaultActor(BaseActor):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        # pylint: disable-next=non-parent-init-called
        nn.Module.__init__(self)

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.use_avg_l1_norm = True

        zs_dim = 256
        hidden_sizes = [256, 256]

        self.feature_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
        )

        self.act_body = nn.Sequential(
            nn.Linear(zs_dim + hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        )

        self.act_head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Tanh(),
        )


class Actor(BaseActor):
    def __init__(
        self, observation_size: int, num_actions: int, config: TD7SIGRegConfig
    ):
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, config=config
        )
