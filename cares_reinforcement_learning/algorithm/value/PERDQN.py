"""
Original Paper: https://arxiv.org/abs/1706.10295
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.PERDQN import Network
from cares_reinforcement_learning.util.configurations import PERDQNConfig


class PERDQN(DQN):
    def __init__(
        self,
        network: Network,
        config: PERDQNConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)
