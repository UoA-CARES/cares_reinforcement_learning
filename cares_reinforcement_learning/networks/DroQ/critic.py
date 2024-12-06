from torch import nn

from cares_reinforcement_learning.networks.SAC import BaseCritic
from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import DroQConfig


# Default network should have this architecture with hidden_sizes = [256, 256]:
class DefaultCritic(BaseCritic):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        input_size = observation_size + num_actions
        hidden_sizes = [256, 256]

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        Q2 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q1=Q1, Q2=Q2)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: DroQConfig):
        input_size = observation_size + num_actions
        hidden_sizes = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        Q2 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(0.005),
            nn.LayerNorm(hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q1=Q1, Q2=Q2)
