from cares_reinforcement_learning.networks.SAC import Critic as SACCritic
from cares_reinforcement_learning.networks.SAC import (
    DefaultCritic as SACDefaultCritic,
)
from cares_reinforcement_learning.algorithm.configurations import MASACConfig


class DefaultCritic(SACDefaultCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: dict, num_actions: int):
        input_size = observation_size["state"]
        num_agents = observation_size["num_agents"]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions * num_agents,
            hidden_sizes=[256, 256],
        )


class Critic(SACCritic):
    def __init__(self, observation_size: dict, num_actions: int, config: MASACConfig):
        input_size = observation_size["state"]
        num_agents = observation_size["num_agents"]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions * num_agents,
            config=config,
        )
