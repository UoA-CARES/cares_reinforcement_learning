"""
Original Paper:
"""

import torch

from cares_reinforcement_learning.algorithm.value import C51
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.Rainbow import Network
from cares_reinforcement_learning.util.configurations import RainbowConfig


class Rainbow(C51):
    def __init__(
        self,
        network: Network,
        config: RainbowConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

    def reset_noise(self):
        self.network.reset_noise()
        self.target_network.reset_noise()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict:
        info = super().train_policy(memory, batch_size)
        self.reset_noise()
        return info
