"""
Rainbow IQN
-----------

Combines the Rainbow DQN extensions with Implicit Quantile Networks.

RainbowIQN = IQN +
             Double Q-learning +
             Dueling architecture +
             Prioritized Experience Replay +
             Multi-step returns +
             Noisy Networks

IQN replaces Rainbow's categorical C51 distribution with a
continuous quantile function.

The quantile regression objective is inherited directly from IQN.
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.configurations import RainbowIQNConfig
from cares_reinforcement_learning.algorithm.value.IQN import IQN
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.RainbowIQN import Network
from cares_reinforcement_learning.types.episode import EpisodeContext


class RainbowIQN(IQN):
    network: Network
    target_network: Network

    def __init__(
        self,
        network: Network,
        config: RainbowIQNConfig,
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
