import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import SIMBAConfig
from cares_reinforcement_learning.networks.common import BasePolicy, TanhGaussianPolicy


class ResidualSequential(nn.Sequential):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + super().forward(x)


class DefaultActor(TanhGaussianPolicy):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: int = 128,
        log_std_bounds: list[float] | None = None,
    ):
        if log_std_bounds is None:
            log_std_bounds = [-10.0, 2.0]

        BasePolicy.__init__(self, observation_size, num_actions)

        self.log_std_bounds = log_std_bounds

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            ResidualSequential(
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size),
            ),
            nn.LayerNorm(hidden_size),
        )

        self.mean_linear = nn.Linear(hidden_size, num_actions)
        self.log_std_linear = nn.Linear(hidden_size, num_actions)


class Actor(TanhGaussianPolicy):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: SIMBAConfig,
    ):
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            log_std_bounds=config.log_std_bounds,
            config=config.actor_config,
        )
