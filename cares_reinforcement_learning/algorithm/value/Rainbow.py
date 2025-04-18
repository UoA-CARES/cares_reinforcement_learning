"""
Original Paper:
"""

from typing import Any

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

    def _reset_noise(self) -> None:
        self.network.reset_noise()
        self.target_network.reset_noise()

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        info = super().train_policy(memory, batch_size, training_step)
        self._reset_noise()
        return info
