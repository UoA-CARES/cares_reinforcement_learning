"""
Original Paper:
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.value import C51
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.networks.Rainbow import Network
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import RainbowConfig


class Rainbow(C51):
    network: Network
    target_network: Network

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
        self,
        memory_buffer: MemoryBuffer[SARLObservation],
        training_context: EpisodeContext,
    ) -> dict[str, Any]:
        info = super().train_policy(memory_buffer, training_context)
        self._reset_noise()
        return info
