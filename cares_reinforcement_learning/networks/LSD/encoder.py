from torch import nn
import torch

from cares_reinforcement_learning.algorithm.configurations import LSDConfig
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseEncoder(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the discriminator network.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: State latent
        """
        return self.network(state)


class DefaultEncoder(BaseEncoder):
    def __init__(self, observation_size: int):

        network = nn.Sequential(
            nn.Linear(in_features=observation_size, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=2),
        )

        super().__init__(network=network)


class Encoder(BaseEncoder):
    def __init__(self, observation_size: int, config: LSDConfig):

        network = MLP(
            input_size=observation_size,
            output_size=config.skill_dim,
            config=config.encoder_config,
        )

        super().__init__(network=network)
