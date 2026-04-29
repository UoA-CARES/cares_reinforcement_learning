from typing import Any

import torch
from torch import nn

from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.networks.noisy_linear import NoisyLinear
from cares_reinforcement_learning.algorithm.configurations import NoisyNetConfig


class BaseNoisyNetwork(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, network: nn.Module):
        super().__init__(observation_size=observation_size, num_actions=num_actions)
        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def reset_noise(self):
        for module in self.network.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    def noise_stats(self) -> dict[str, Any]:
        return self._module_noise_stats("network", self.network)


class DefaultNetwork(BaseNoisyNetwork):
    def __init__(self, observation_size: int, num_actions: int):
        network = nn.Sequential(
            nn.Linear(observation_size, 64),
            nn.ReLU(),
            NoisyLinear(64, 64, sigma_init=1.0),
            nn.ReLU(),
            NoisyLinear(64, num_actions, sigma_init=0.5),
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )


class Network(BaseNoisyNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: NoisyNetConfig):

        network = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.network_config,
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )
