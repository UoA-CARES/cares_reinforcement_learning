import torch
from torch import nn
from torchrl.modules import NoisyLinear
from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import NoisyNetConfig


class BaseNetwork(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()
        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

    def reset_noise(self):
        for module in self.network.children():
            if hasattr(module, "reset_noise"):
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

        network = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.network_config,
        )
        super().__init__(network=network)
