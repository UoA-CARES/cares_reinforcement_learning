from torch import nn

from cares_reinforcement_learning.util.common import ContinuousDistributedCritic
from cares_reinforcement_learning.util.configurations import CTD4Config, MLPConfig


# This is the default base network for CTD4 for reference and testing of default network configurations
class DefaultCritic(ContinuousDistributedCritic):
    def __init__(self, observation_size: int, action_num: int):
        input_size = observation_size + action_num
        hidden_sizes = [256, 256]

        super().__init__(
            input_size=input_size,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.std_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.soft_std_layer = nn.Softplus()


class Critic(ContinuousDistributedCritic):
    def __init__(self, observation_size: int, action_num: int, config: CTD4Config):
        input_size = observation_size + action_num

        super().__init__(input_size=input_size, config=config.critic_config)
