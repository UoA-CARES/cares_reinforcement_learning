from cares_reinforcement_learning.algorithm.configurations import SIMBAConfig
from cares_reinforcement_learning.networks.common import TwinQNetwork


class Critic(TwinQNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: SIMBAConfig,
    ):
        super().__init__(
            input_size=observation_size + num_actions,
            output_size=1,
            config=config.critic_config,
        )
