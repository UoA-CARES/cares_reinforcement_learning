import torch
from torch import nn

from cares_reinforcement_learning.networks.CTD4 import DistributedCritic as Critic


class EnsembleCritic(nn.ModuleList):
    def __init__(
        self,
        ensemble_size: int,
        observation_size: int,
        action_num: int,
        hidden_size: list[int],
    ):
        super().__init__()

        self.ensemble_size = ensemble_size

        critics = [
            Critic(observation_size, action_num, hidden_size=hidden_size)
            for _ in range(self.ensemble_size)
        ]
        self.extend(critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        return [critic(state, action) for critic in self]
