"""
Rainbow DQN
------------

Original Paper: https://arxiv.org/pdf/1710.02298

Rainbow integrates multiple DQN improvements into a single,
unified algorithm for discrete control.

Core Idea:
- Combine complementary advances to improve stability,
  sample efficiency, and final performance.

Rainbow = DQN +
           Double Q-learning +
           Dueling architecture +
           Prioritized Experience Replay +
           Multi-step returns +
           Distributional RL (C51) +
           Noisy Networks

Components:

1) Double DQN
   - Decouples action selection and evaluation
   - Reduces overestimation bias.

2) Dueling Network
   - Separates value and advantage streams
   - Improves representation learning.

3) Prioritized Replay (PER)
   - Samples transitions ∝ TD-error
   - Uses importance sampling weights.

4) Multi-step Returns
   - Uses n-step targets:
         y = r₀ + γ r₁ + ... + γⁿ Q(sₙ, a*)
   - Speeds reward propagation.

5) Distributional RL (C51)
   - Learns categorical return distribution
   - Uses cross-entropy loss on projected atoms.

6) Noisy Nets
   - Replaces ε-greedy with learned parameter noise.

Training:
- Uses replay buffer with PER.
- Target network for stability.
- Categorical Bellman projection.
- All components integrated in a single loss.

Key Behaviour:
- Significantly improves sample efficiency.
- Strong Atari benchmark performance.
- Demonstrates additive benefits of DQN extensions.

Rainbow = a unified, multi-component enhancement
           of the original DQN framework.
"""

from typing import Any

import torch

from cares_reinforcement_learning.algorithm.value import C51
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.Rainbow import Network
from cares_reinforcement_learning.types.episode import EpisodeContext
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

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info = super().train(memory_buffer, episode_context)
        self._reset_noise()
        return info
