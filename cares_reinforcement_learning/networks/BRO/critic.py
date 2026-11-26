"""BRO twin quantile critics."""

from torch import nn

from cares_reinforcement_learning.algorithm.configurations import BROConfig
from cares_reinforcement_learning.networks.bro_architecture import (
    BroNetBlock,
    bro_weight_init,
)
from cares_reinforcement_learning.networks.common import (
    BaseCritic,
    TwinQNetwork,
)


class DefaultCritic(TwinQNetwork):
    """Default BRO critic: width 512, two BroNet blocks, 100 quantiles."""

    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        num_quantiles = 100

        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=num_quantiles,
        )

        def q_network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_size, 512),
                nn.LayerNorm(512, eps=1e-6),
                nn.ReLU(),
                BroNetBlock(512),
                BroNetBlock(512),
                nn.Linear(512, num_quantiles),
            )

        self.Q1 = q_network()
        self.Q2 = q_network()

        self.apply(bro_weight_init)


class Critic(TwinQNetwork):
    """Configurable BRO twin quantile critics."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: BROConfig,
    ):
        super().__init__(
            input_size=observation_size + num_actions,
            output_size=config.num_quantiles,
            config=config.critic_config,
        )

        self.apply(bro_weight_init)
