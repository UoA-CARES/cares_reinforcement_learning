from cares_reinforcement_learning.algorithm.configurations import SIMBAConfig
from cares_reinforcement_learning.networks.common import TanhGaussianPolicy


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
