import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.algorithm.configurations import CICConfig

class BaseStateEncoder(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class DefaultStateEncoder(BaseStateEncoder):
    def __init__(self, observation_size:int):
        
        network = nn.Sequential(
            nn.Linear(in_features=observation_size, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=64)
        )

        super().__init__(network=network)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)

class StateEncoder(BaseStateEncoder):
    def __init__(self, observation_size: int, config: CICConfig):

        network = MLP(
            input_size=observation_size,
            output_size=config.z_dim,
            config=config.state_encoder_config
        )

        super().__init__(network=network)
