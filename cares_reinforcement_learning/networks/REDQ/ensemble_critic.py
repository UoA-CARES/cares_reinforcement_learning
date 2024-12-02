import torch
from torch import nn

from cares_reinforcement_learning.networks.REDQ import DefaultCritic
from cares_reinforcement_learning.networks.REDQ import Critic
from cares_reinforcement_learning.util.configurations import REDQConfig


# TODO merge this into CTD4 Critic similar to TQC and REDQ
class BaseEnsembleCritic(nn.ModuleList):
    def __init__(self, critics: list[Critic] | list[DefaultCritic]):
        super().__init__()

        self.extend(critics)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> list[torch.Tensor]:
        return [critic(state, action) for critic in self]


# This is the default base network for CTD4 for reference and testing of default network configurations
class DefaultEnsembleCritic(BaseEnsembleCritic):
    def __init__(self, observation_size: int, action_num: int):

        ensemble_size = 10

        critics = [
            DefaultCritic(observation_size, action_num) for _ in range(ensemble_size)
        ]

        super().__init__(critics=critics)


class EnsembleCritic(BaseEnsembleCritic):
    def __init__(
        self,
        observation_size: int,
        action_num: int,
        config: REDQConfig,
    ):
        ensemble_size = config.ensemble_size

        critics = [
            Critic(observation_size, action_num, config=config)
            for _ in range(ensemble_size)
        ]
        super().__init__(critics=critics)
