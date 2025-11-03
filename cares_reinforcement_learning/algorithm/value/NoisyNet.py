"""
Original Paper: https://arxiv.org/abs/1706.10295
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.NoisyNet import Network
from cares_reinforcement_learning.util.configurations import NoisyNetConfig
from cares_reinforcement_learning.util.training_context import TrainingContext


class NoisyNet(DQN):
    def __init__(
        self,
        network: Network,
        config: NoisyNetConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

    def _reset_noise(self):
        self.network.reset_noise()
        self.target_network.reset_noise()

    def train_policy(self, training_context: TrainingContext) -> dict:
        info = super().train_policy(training_context)
        self._reset_noise()
        return info
