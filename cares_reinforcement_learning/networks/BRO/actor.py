"""BRO Gaussian actor and optimistic exploration actor."""

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import BROConfig
from cares_reinforcement_learning.networks.bro_architecture import (
    BroNetBlock,
    bro_weight_init,
)
from cares_reinforcement_learning.networks.common import (
    BasePolicy,
    TanhGaussianPolicy,
)


class BaseActor(TanhGaussianPolicy):
    """BRO exploitation policy."""

    def gaussian_parameters(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return the unsquashed Gaussian mean and standard deviation.

        Uses the same parameterization as TanhGaussianPolicy.forward().
        """
        features = self.act_net(state)

        mean = self.mean_linear(features)
        log_std = self.log_std_linear(features)

        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)

        return mean, log_std.exp()


class DefaultActor(BaseActor):
    """Default BRO actor: width 256, one BroNet block."""

    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        BasePolicy.__init__(self, observation_size, num_actions)

        self.log_std_bounds = [-10.0, 2.0]

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.ReLU(),
            BroNetBlock(256),
        )

        self.mean_linear = nn.Linear(256, num_actions)
        self.log_std_linear = nn.Linear(256, num_actions)

        self.apply(bro_weight_init)


class Actor(BaseActor):
    """Configurable BRO exploitation actor."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: BROConfig,
    ):
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            log_std_bounds=config.log_std_bounds,
            config=config.actor_config,
        )

        self.apply(bro_weight_init)
