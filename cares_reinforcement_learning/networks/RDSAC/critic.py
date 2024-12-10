from torch import nn

from cares_reinforcement_learning.util.common import TwinQValueCritic
from cares_reinforcement_learning.util.configurations import RDSACConfig, MLPConfig


class DefaultCritic(TwinQValueCritic):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        hidden_sizes = [256, 256]
        output_size = 1 + 1 + observation_size

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )


class Critic(TwinQValueCritic):
    def __init__(self, observation_size: int, num_actions: int, config: RDSACConfig):
        input_size = observation_size + num_actions
        output_size = 1 + 1 + observation_size

        super().__init__(
            input_size=input_size, output_size=output_size, config=config.critic_config
        )
