"""
This is a stub file for the Critic class - reads directly off SAC's Critic class.
"""

# pylint: disable=unused-import
from torch import nn

from cares_reinforcement_learning.networks.common import BaseCritic, TwinQNetwork
from cares_reinforcement_learning.networks.SAC import Critic
from cares_reinforcement_learning.util.configurations import DroQConfig


# pylint: disable=super-init-not-called
class DefaultCritic(TwinQNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        input_size = observation_size + num_actions
        hidden_sizes = [256, 256]

        # pylint: disable-next=non-parent-init-called
        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=1,
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
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
        self.Q2 = nn.Sequential(
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
