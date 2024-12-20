from torch import nn

from cares_reinforcement_learning.networks.common import QNetwork
from cares_reinforcement_learning.util.configurations import DDPGConfig, MLPConfig


class DefaultCritic(QNetwork):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        hidden_sizes = [1024, 1024]

        super().__init__(
            input_size=input_size,
            output_size=1,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        # Q architecture
        # pylint: disable-next=invalid-name
        self.Q = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )


class Critic(QNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: DDPGConfig):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size, output_size=1, config=config.critic_config
        )
