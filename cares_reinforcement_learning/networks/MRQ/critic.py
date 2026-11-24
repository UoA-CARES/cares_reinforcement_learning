"""MR.Q twin critic heads taking only z_sa as input."""

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import MRQConfig
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseCritic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: MRQConfig):
        super().__init__()
        self.input_size = config.zsa_dim
        self.output_size = 1
        self.Q1: MLP | nn.Sequential = MLP(
            input_size=config.zsa_dim, output_size=1, config=config.critic_config
        )
        self.Q2: MLP | nn.Sequential = MLP(
            input_size=config.zsa_dim, output_size=1, config=config.critic_config
        )

    def forward(self, zsa: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.Q1(zsa), self.Q2(zsa)


class DefaultCritic(BaseCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.input_size = 512
        self.output_size = 1

        def q_network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(512, 512),
                nn.LayerNorm(512, elementwise_affine=False),
                nn.ELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512, elementwise_affine=False),
                nn.ELU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512, elementwise_affine=False),
                nn.ELU(),
                nn.Linear(512, 1),
            )

        self.Q1 = q_network()
        self.Q2 = q_network()


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: MRQConfig):
        super().__init__(observation_size, num_actions, config)
