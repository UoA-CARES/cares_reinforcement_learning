import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.algorithm.configurations import CICConfig

class BaseTransitionEncoder(nn.Module):
    def __init__(self, network: nn.Module):
        super().__init__()

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.network(state)
    
class DefaultTransitionEncoder(BaseTransitionEncoder):
    def __init__(self):
        
        network = nn.Sequential(
            nn.Linear(in_features=64*2, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024,out_features=64)
        )

        super().__init__(network)

class TransitionEncoder(BaseTransitionEncoder):
    def __init__(self, config:CICConfig):
        network = MLP (
            input_size=2*config.z_dim,
            output_size=config.z_dim,
            config=config.transition_encoder_config
        )

        super().__init__(network)