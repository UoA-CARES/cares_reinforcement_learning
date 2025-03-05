"""
Original paper https://arxiv.org/abs/1511.06581
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.DuelingDQN import Network
from cares_reinforcement_learning.util.configurations import DuelingDQNConfig


class DuelingDQN(DQN):
    def __init__(
        self,
        network: Network,
        config: DuelingDQNConfig,
        device: torch.device,
    ):
        super().__init__(network, config, device)
