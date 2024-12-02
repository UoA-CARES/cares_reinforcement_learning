from torch import nn

from cares_reinforcement_learning.networks.SAC import BaseCritic
from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import MAPERSACConfig


class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        hidden_sizes = [400, 300]
        output_size = 1 + 1 + observation_size

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q2 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )
        super().__init__(Q1=Q1, Q2=Q2)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: MAPERSACConfig):
        input_size = observation_size + num_actions
        hidden_sizes = config.hidden_size_critic
        output_size = 1 + 1 + observation_size

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = MLP(
            input_size,
            hidden_sizes,
            output_size=output_size,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        Q2 = MLP(
            input_size,
            hidden_sizes,
            output_size=output_size,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )
        super().__init__(Q1=Q1, Q2=Q2)
