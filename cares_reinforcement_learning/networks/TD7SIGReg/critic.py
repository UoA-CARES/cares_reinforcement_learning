"""
TD7SIGReg critic - TD7's twin critic with each Q MLP split into an explicit
body (ending in a Linear layer) and head (activation + output layer).

This exposes two representations per twin for the optional SIGReg regulariser:
    feature     - feature-layer output (the tensor fed to AvgL1Norm, pre-norm)
    penultimate - body output: pre-activation output of the last hidden Linear

AvgL1Norm on the feature layers is optional (config.critic_use_avg_l1_norm).
"""

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import TD7SIGRegConfig
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseCritic(nn.Module):
    def __init__(
        self, observation_size: int, num_actions: int, config: TD7SIGRegConfig
    ):
        super().__init__()

        self.input_size = observation_size + num_actions
        self.output_size = 1
        self.use_avg_l1_norm = bool(config.critic_use_avg_l1_norm)

        # Q1 architecture
        self.feature_layer_one: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        q1_input_size = 2 * config.zs_dim + self.feature_layer_one.output_size

        # pylint: disable-next=invalid-name
        self.Q1_body: MLP | nn.Sequential = MLP(
            input_size=q1_input_size,
            output_size=None,
            config=config.critic_body_config,
        )
        # pylint: disable-next=invalid-name
        self.Q1_head: MLP | nn.Sequential = MLP(
            input_size=self.Q1_body.output_size,
            output_size=1,
            config=config.critic_head_config,
        )

        # Q2 architecture
        self.feature_layer_two: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        q2_input_size = 2 * config.zs_dim + self.feature_layer_two.output_size

        # pylint: disable-next=invalid-name
        self.Q2_body: MLP | nn.Sequential = MLP(
            input_size=q2_input_size,
            output_size=None,
            config=config.critic_body_config,
        )
        # pylint: disable-next=invalid-name
        self.Q2_head: MLP | nn.Sequential = MLP(
            input_size=self.Q2_body.output_size,
            output_size=1,
            config=config.critic_head_config,
        )

    def _q_forward(
        self,
        feature_layer: nn.Module,
        body: nn.Module,
        head: nn.Module,
        obs_action: torch.Tensor,
        embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        feature = feature_layer(obs_action)
        latent = fnc.avg_l1_norm(feature) if self.use_avg_l1_norm else feature
        latent = torch.cat([latent, embeddings], dim=1)
        penultimate = body(latent)
        q = head(penultimate)
        return q, feature, penultimate

    def forward_with_features(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        zsa: torch.Tensor,
        zs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, list[torch.Tensor]]]:
        obs_action = torch.cat([state, action], dim=1)
        embeddings = torch.cat([zsa, zs], dim=1)

        q1, feature_one, penultimate_one = self._q_forward(
            self.feature_layer_one, self.Q1_body, self.Q1_head, obs_action, embeddings
        )
        q2, feature_two, penultimate_two = self._q_forward(
            self.feature_layer_two, self.Q2_body, self.Q2_head, obs_action, embeddings
        )

        features = {
            "feature": [feature_one, feature_two],
            "penultimate": [penultimate_one, penultimate_two],
        }
        return q1, q2, features

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        zsa: torch.Tensor,
        zs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, q2, _ = self.forward_with_features(state, action, zsa, zs)
        return q1, q2


# Reference network (TD7 critic layers, split into body + head) for testing
class DefaultCritic(BaseCritic):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        input_size = observation_size + num_actions
        zs_dim = 256

        # pylint: disable-next=non-parent-init-called
        nn.Module.__init__(self)

        self.input_size = input_size
        self.output_size = 1
        self.use_avg_l1_norm = True

        q_input_size = 2 * zs_dim + hidden_sizes[0]

        self.feature_layer_one = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
        )
        # pylint: disable-next=invalid-name
        self.Q1_body = nn.Sequential(
            nn.Linear(q_input_size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        )
        # pylint: disable-next=invalid-name
        self.Q1_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.feature_layer_two = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
        )
        # pylint: disable-next=invalid-name
        self.Q2_body = nn.Sequential(
            nn.Linear(q_input_size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
        )
        # pylint: disable-next=invalid-name
        self.Q2_head = nn.Sequential(
            nn.ELU(),
            nn.Linear(hidden_sizes[1], 1),
        )


class Critic(BaseCritic):
    def __init__(
        self, observation_size: int, num_actions: int, config: TD7SIGRegConfig
    ):
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )
