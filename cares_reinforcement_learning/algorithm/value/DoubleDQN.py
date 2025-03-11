"""
Original Paper: https://arxiv.org/abs/1509.06461

code based on: https://github.com/dxyang/DQN_pytorch
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.DoubleDQN import Network as DoubleDQNNetwork
from cares_reinforcement_learning.networks.DuelingDQN import (
    Network as DuelingDQNNetwork,
)
from cares_reinforcement_learning.util.configurations import DoubleDQNConfig


class DoubleDQN(DQN):
    def __init__(
        self,
        network: DoubleDQNNetwork | DuelingDQNNetwork,
        config: DoubleDQNConfig,
        device: torch.device,
    ):
        super().__init__(network, config, device)
