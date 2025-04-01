import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import QRDQNConfig


class BaseNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        quantiles: int,
        network: nn.Module | nn.Sequential,
    ):
        super().__init__()

        self.input_size = input_size

        self.num_actions = num_actions
        self.quantiles = quantiles

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        quantiles = self.calculate_quantiles(state)

        return quantiles.mean(dim=-1)

    def calculate_quantiles(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)

        return output.view(
            state.shape[0],
            self.num_actions,
            self.quantiles,
        )


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        quantiles = 200
        hidden_sizes = [256, 256]

        network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions * quantiles),
        )
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            quantiles=quantiles,
            network=network,
        )


class Network(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: QRDQNConfig):

        network = MLP(
            input_size=observation_size,
            output_size=num_actions * config.quantiles,
            config=config.network_config,
        )
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            quantiles=config.quantiles,
            network=network,
        )
