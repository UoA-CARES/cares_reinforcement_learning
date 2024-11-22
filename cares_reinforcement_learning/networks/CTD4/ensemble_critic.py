import torch
from torch import nn

from cares_reinforcement_learning.networks.CTD4 import DistributedCritic as Critic
from cares_reinforcement_learning.util.configurations import CTD4Config


class EnsembleCritic(nn.ModuleList):
    def __init__(
        self,
        observation_size: int,
        action_num: int,
        config: CTD4Config,
    ):
        super().__init__()

        self.hidden_sizes = config.hidden_size_critic
        self.ensemble_size = config.ensemble_size

        critics = [
            Critic(observation_size, action_num, config=config)
            for _ in range(self.ensemble_size)
        ]
        self.extend(critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        return [critic(state, action) for critic in self]
