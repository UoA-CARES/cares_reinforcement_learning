"""
TD7SIGReg encoder - identical to TD7's SALE encoder except the state embedding
zs is NOT normalised with AvgL1Norm. Embedding scale is instead regularised
by the SIGReg loss term in the encoder objective.
"""

import torch

from cares_reinforcement_learning.algorithm.configurations import TD7SIGRegConfig
from cares_reinforcement_learning.networks.TD7.encoder import (
    BaseEncoder as TD7BaseEncoder,
)
from cares_reinforcement_learning.networks.TD7.encoder import (
    DefaultEncoder as TD7DefaultEncoder,
)


class BaseEncoder(TD7BaseEncoder):
    def zs(self, state: torch.Tensor) -> torch.Tensor:
        # No AvgL1Norm - SIGReg regularises the embedding distribution instead
        return self.state_encoder(state)


class DefaultEncoder(TD7DefaultEncoder):
    def zs(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state)


class Encoder(BaseEncoder):
    def __init__(
        self, observation_size: int, num_actions: int, config: TD7SIGRegConfig
    ):
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )
