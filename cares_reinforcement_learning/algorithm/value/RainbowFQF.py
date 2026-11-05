"""
Rainbow FQF
-----------

Combines the Rainbow DQN extensions with Fully Parameterized
Quantile Functions.

RainbowFQF = FQF +
             Double Q-learning +
             Dueling architecture +
             Prioritized Experience Replay +
             Multi-step returns +
             Noisy Networks

The quantile regression and fraction proposal objectives
are inherited directly from FQF.
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.configurations import RainbowFQFConfig
from cares_reinforcement_learning.algorithm.value.FQF import FQF
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.RainbowFQF import Network
from cares_reinforcement_learning.types.episode import EpisodeContext


class RainbowFQF(FQF):
    network: Network
    target_network: Network

    def __init__(
        self,
        network: Network,
        config: RainbowFQFConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

    def _reset_noise(self) -> None:
        self.network.reset_noise()
        self.target_network.reset_noise()

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info = super().train(memory_buffer, episode_context)
        info.update(self.network.noise_stats())
        self._reset_noise()
        return info
