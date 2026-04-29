"""
Dueling DQN (Dueling Deep Q-Network)
--------------------------------------

Original paper https://arxiv.org/abs/1511.06581

Dueling DQN modifies the Q-network architecture to separately
estimate state value and action advantage, improving learning
efficiency in environments where many actions have similar value.

Core Problem:
- In many states, the choice of action has little effect.
- Standard DQN must learn Q(s,a) for each action separately.
- This can slow learning and waste capacity.

Core Idea:
- Decompose Q(s,a) into:
      Q(s,a) = V(s) + A(s,a)

where:
    V(s)      = state value
    A(s,a)    = advantage of action a relative to others

To ensure identifiability, advantages are normalized:

    Q(s,a) = V(s) + ( A(s,a) - mean_a A(s,a) )

Architecture:
- Shared feature extractor.
- Two output streams:
      • Value stream → V(s)
      • Advantage stream → A(s,a)
- Combined to produce Q-values.

Training:
- Same DQN loss:
      y = r + γ (1 - done) max_a Q_target(s', a)
      L = (Q(s,a) - y)^2
- Replay buffer and target network unchanged.

Key Behaviour:
- Learns state value even when actions are similar.
- Improves stability and sample efficiency.
- Especially helpful in large action spaces.

Dueling DQN = DQN with separate value and advantage streams.
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.DuelingDQN import Network
from cares_reinforcement_learning.algorithm.configurations import DuelingDQNConfig


class DuelingDQN(DQN):
    def __init__(
        self,
        network: Network,
        config: DuelingDQNConfig,
        device: torch.device,
    ):
        super().__init__(network, config, device)
