from cares_reinforcement_learning.networks.TD3 import Critic as TD3Critic
from cares_reinforcement_learning.networks.TD3 import (
    DefaultCritic as TD3DefaultCritic,
)
from cares_reinforcement_learning.algorithm.configurations import MATD3Config


class DefaultCritic(TD3DefaultCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: dict, num_actions: int):
        input_size = observation_size["state"]
        num_agents = observation_size["num_agents"]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions * num_agents,
            hidden_sizes=[256, 256],
        )


class Critic(TD3Critic):
    def __init__(self, observation_size: dict, num_actions: int, config: MATD3Config):
        input_size = observation_size["state"]
        num_agents = observation_size["num_agents"]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions * num_agents,
            config=config,
        )
