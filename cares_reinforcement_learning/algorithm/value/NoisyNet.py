"""
Original Paper: https://arxiv.org/abs/1706.10295
"""

import torch
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.algorithm.value.DQN import DQN
from cares_reinforcement_learning.networks.NoisyNet import network as NoisyNetwork
from cares_reinforcement_learning.util.configurations import NoisyNetConfig


class NoisyNet(DQN):
    def __init__(
        self,
        network: NoisyNetwork,
        config: NoisyNetConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict:
        info = super().train_policy(memory, batch_size)
        self.network.reset_noise()
        return info
