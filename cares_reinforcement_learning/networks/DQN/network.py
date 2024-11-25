import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import DQNConfig


class BaseNetwork(nn.Module):
    def __init__(
        self,
        network: nn.Module,
    ):
        super().__init__()

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)
        return output


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [512, 512]

        network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )
        super().__init__(network=network)


class Network(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: DQNConfig):
        hidden_sizes = config.hidden_size

        network = MLP(
            observation_size,
            hidden_sizes,
            output_size=num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )
        super().__init__(network=network)
