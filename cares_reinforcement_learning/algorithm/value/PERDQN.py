"""
PER-DQN (Prioritized Experience Replay DQN)
--------------------------------------------

Original Paper: https://arxiv.org/abs/1706.10295

PER-DQN extends DQN by sampling replay transitions
non-uniformly based on their learning importance.

Core Problem:
- Uniform replay treats all transitions equally.
- Many transitions have small TD-error and contribute
  little to learning.
- Sample efficiency can be improved by focusing on
  informative experiences.

Core Idea:
- Assign each transition a priority based on TD-error:
      δ_i = Q(s,a) - y
- Sampling probability:
      p_i ∝ (|δ_i| + ε)^α
  where α controls prioritization strength.

Importance Sampling Correction:
- Because sampling is no longer uniform,
  apply importance weights to correct bias:
      w_i = (1 / (N p_i))^β
- Critic loss becomes:
      L = w_i * (Q(s,a) - y)^2
- β is annealed toward 1 over training.

Target (unchanged from DQN):
      y = r + γ (1 - done) max_a Q_target(s', a)

Key Behaviour:
- High TD-error transitions replayed more often.
- Faster propagation of value information.
- Slight bias introduced but corrected via IS weights.

Advantages:
- Improved sample efficiency over uniform replay.
- Simple modification to DQN framework.

PER-DQN = DQN + prioritized replay + importance weighting.
"""

import torch

from cares_reinforcement_learning.algorithm.value import DQN
from cares_reinforcement_learning.networks.PERDQN import Network
from cares_reinforcement_learning.util.configurations import PERDQNConfig


class PERDQN(DQN):
    def __init__(
        self,
        network: Network,
        config: PERDQNConfig,
        device: torch.device,
    ):
        super().__init__(network=network, config=config, device=device)
