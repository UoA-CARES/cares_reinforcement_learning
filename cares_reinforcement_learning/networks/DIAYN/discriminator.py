import torch
from torch import nn
from torch.nn import functional as F

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import DIAYNConfig


class BaseDiscriminator(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output logits from the discriminator.
        """
        return self.network(state)


class DefaultDiscriminator(BaseDiscriminator):
    def __init__(self, observation_size: int, num_skills: int):

        network = nn.Sequential(
            nn.Linear(in_features=observation_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=num_skills),
        )

        super().__init__(network=network)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output logits from the discriminator.
        """
        return self.network(state)


class Discriminator(BaseDiscriminator):
    def __init__(self, observation_size: int, num_skills: int, config: DIAYNConfig):
        network = MLP(
            input_size=observation_size,
            output_size=num_skills,
            config=config.discriminator_config,
        )

        super().__init__(network=network)
