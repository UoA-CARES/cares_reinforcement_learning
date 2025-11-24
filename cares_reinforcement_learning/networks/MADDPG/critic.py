from cares_reinforcement_learning.networks.DDPG import Critic as DDPGCritic
from cares_reinforcement_learning.networks.DDPG import (
    DefaultCritic as DDPGDefaultCritic,
)
from cares_reinforcement_learning.util.configurations import MADDPGConfig


class DefaultCritic(DDPGDefaultCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: dict, num_actions: int):
        input_size = observation_size["state"]
        num_agents = observation_size["num_agents"]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions * num_agents,
            hidden_sizes=[256, 256],
        )


class Critic(DDPGCritic):
    def __init__(self, observation_size: dict, num_actions: int, config: MADDPGConfig):
        input_size = observation_size["state"]
        num_agents = observation_size["num_agents"]

        super().__init__(
            observation_size=input_size,
            num_actions=num_actions * num_agents,
            config=config,
        )
