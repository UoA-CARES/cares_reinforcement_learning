"""
NoisyNet (Noisy Networks for Exploration)
------------------------------------------

Original Paper: https://arxiv.org/abs/1706.10295

NoisyNet replaces deterministic linear layers with
parameterized stochastic layers to enable learned exploration.

Core Problem:
- ε-greedy exploration is:
      • State-independent
      • Manually scheduled
      • Inefficient in sparse-reward settings
- Exploration should adapt to learning progress.

Core Idea:
- Inject trainable, factorized Gaussian noise directly
  into network weights:

      w = μ + σ ⊙ ε

  where:
      μ = learned mean parameters
      σ = learned noise scale parameters
      ε = random noise

- Both μ and σ are optimized via gradient descent.
- Exploration emerges from learned stochasticity.

Noisy Linear Layer:
    y = (μ_w + σ_w ⊙ ε_w) x
        + (μ_b + σ_b ⊙ ε_b)

Factorized Gaussian noise reduces computation cost.

Training:
- Noise is resampled at each forward pass.
- Gradients flow through μ and σ.
- No separate ε-greedy policy needed.

Key Behaviour:
- Exploration becomes state-dependent.
- Noise magnitude shrinks as learning converges.
- Often removes need for ε scheduling entirely.

Advantages:
- More structured exploration than ε-greedy.
- Improved performance in Atari and sparse tasks.
- Simple architectural modification.

NoisyNet = neural networks with learnable parameter noise
           for adaptive exploration.
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.NoisyNet import BaseNoisyNetwork
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.util.configurations import NoisyNetConfig


class NoisyNet(DQN):
    network: BaseNoisyNetwork
    target_network: BaseNoisyNetwork

    def __init__(
        self,
        network: BaseNoisyNetwork,
        config: NoisyNetConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)

    def _reset_noise(self):
        self.network.reset_noise()
        self.target_network.reset_noise()

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict:
        info = super().train(memory_buffer, episode_context)
        info.update(self.network.noise_stats())
        self._reset_noise()
        return info
