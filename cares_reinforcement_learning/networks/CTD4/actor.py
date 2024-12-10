# pylint: disable=unused-import
from cares_reinforcement_learning.networks.TD3 import DefaultActor
from cares_reinforcement_learning.util.common import DeterministicPolicy
from cares_reinforcement_learning.util.configurations import CTD4Config


class Actor(DeterministicPolicy):
    def __init__(self, observation_size: int, num_actions: int, config: CTD4Config):

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            config=config.actor_config,
        )
