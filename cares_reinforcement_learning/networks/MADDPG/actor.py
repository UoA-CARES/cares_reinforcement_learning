from torch import nn

from cares_reinforcement_learning.networks.common import DeterministicPolicy, BasePolicy
from cares_reinforcement_learning.util.configurations import DDPGConfig


class DefaultActor(DeterministicPolicy):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        hidden_sizes = [1024, 1024]

        # pylint: disable-next=non-parent-init-called
        BasePolicy.__init__(
            self,
            input_size=observation_size,
            num_actions=num_actions,
        )

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Tanh(),
        )


class Actor(DeterministicPolicy):
    def __init__(self, observation_size: int, num_actions: int, config: DDPGConfig):

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            config=config.actor_config,
        )
