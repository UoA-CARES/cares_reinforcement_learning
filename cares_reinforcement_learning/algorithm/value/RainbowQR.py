"""
Rainbow QR-DQN
--------------

Combines Rainbow DQN extensions with Quantile Regression DQN.

RainbowQR = QRDQN +
            Double Q-learning +
            Dueling architecture +
            Prioritized Experience Replay +
            Multi-step returns +
            Noisy Networks

The fixed-quantile distributional regression objective is inherited
from QRDQN, replacing Rainbow's categorical C51 distribution.
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.configurations import RainbowQRConfig
from cares_reinforcement_learning.algorithm.value.QRDQN import QRDQN
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.RainbowQR import Network
from cares_reinforcement_learning.types.episode import EpisodeContext


class RainbowQR(QRDQN):
    network: Network
    target_network: Network

    def __init__(
        self,
        network: Network,
        config: RainbowQRConfig,
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
