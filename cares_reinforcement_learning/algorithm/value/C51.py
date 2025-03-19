"""
Original Paper:
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.C51 import Network
from cares_reinforcement_learning.util.configurations import C51Config


class C51(DQN):
    def __init__(
        self,
        network: Network,
        config: C51Config,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)
