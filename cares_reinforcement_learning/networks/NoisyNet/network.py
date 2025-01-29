import torch
from torch import nn
from torchrl.modules import NoisyLinear
from cares_reinforcement_learning.util.configurations import NoisyNetConfig


class BaseNetwork(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def reset_noise(self):
        for module in self.network:
            if isinstance(module, NoisyLinear):
                module.reset_noise()


class DefaultNetwork(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int):
        super().__init__(
            nn.Sequential(
                NoisyLinear(observation_size, 512),
                nn.ReLU(),
                NoisyLinear(512, 512),
                nn.ReLU(),
                NoisyLinear(512, num_actions),
            )
        )


# MLP'iffy once proven to learn
class Network(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: NoisyNetConfig):
        super().__init__(
            nn.Sequential(
                NoisyLinear(observation_size, 512),
                nn.ReLU(),
                NoisyLinear(512, 512),
                nn.ReLU(),
                NoisyLinear(512, num_actions),
            )
        )
