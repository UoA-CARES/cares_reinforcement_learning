from cares_reinforcement_learning.algorithm.configurations import MAPPOConfig
from cares_reinforcement_learning.networks.PPO import Critic as PPOCritic
from cares_reinforcement_learning.networks.PPO import (
    DefaultCritic as PPODefaultCritic,
)


class DefaultCritic(PPODefaultCritic):
    def __init__(
        self,
        observation_size: dict,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        input_size = observation_size["state"]

        super().__init__(
            observation_size=input_size,
            hidden_sizes=hidden_sizes,
        )


class Critic(PPOCritic):
    def __init__(self, observation_size: dict, config: MAPPOConfig):
        # Q architecture
        input_size = observation_size["state"]

        # pylint: disable-next=invalid-name
        super().__init__(
            observation_size=input_size,
            config=config,
        )
