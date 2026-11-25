"""MR.Q deterministic latent actor."""

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import MRQConfig
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseActor(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: MRQConfig):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.act_net: MLP | nn.Sequential = MLP(
            input_size=config.zs_dim,
            output_size=num_actions,
            config=config.actor_config,
        )

    def forward(self, zs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre_activation = self.act_net(zs)
        return torch.tanh(pre_activation), pre_activation

    def act(self, zs: torch.Tensor) -> torch.Tensor:
        return self.forward(zs)[0]


class DefaultActor(BaseActor):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.act_net = nn.Sequential(
            nn.Linear(512, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ReLU(),
            nn.Linear(512, num_actions),
        )


class Actor(BaseActor):
    def __init__(self, observation_size: int, num_actions: int, config: MRQConfig):
        super().__init__(observation_size, num_actions, config)
