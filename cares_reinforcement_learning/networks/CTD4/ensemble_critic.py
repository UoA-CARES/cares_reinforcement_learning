import torch
from torch import nn

from cares_reinforcement_learning.networks.CTD4 import DistributedCritic as Critic


class EnsembleCritic(torch.nn.ModuleList):
    def __init__(self, ensemble_size, observation_size, action_num):
        super().__init__()
        self.ensemble_size = ensemble_size

        critics = [Critic(observation_size, action_num) for _ in range(ensemble_size)]
        self.extend(critics)

    def forward(self, state, action):
        return [critic(state, action) for critic in self]
