from torch import nn

from cares_reinforcement_learning.networks.common import DeterministicPolicy
from cares_reinforcement_learning.util.configurations import DDPGConfig, MLPConfig


class DefaultActor(DeterministicPolicy):
    def __init__(self, observation_size: int, num_actions: int):
        hidden_sizes = [1024, 1024]

        actor_config: MLPConfig = MLPConfig(
            hidden_sizes=hidden_sizes, output_activation_function=nn.Tanh.__name__
        )

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            config=actor_config,
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
