import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import DQNConfig


class BaseNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError(
            "BaseDQN is an abstract class and cannot be instantiated directly."
        )


class BaseDQN(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        network: nn.Module,
    ):
        super().__init__(observation_size=observation_size, num_actions=num_actions)

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)
        return output


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseDQN):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [64, 64]

        network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )


class Network(BaseDQN):
    def __init__(self, observation_size: int, num_actions: int, config: DQNConfig):

        network = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.network_config,
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )
