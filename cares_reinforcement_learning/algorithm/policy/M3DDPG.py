"""
M3DDPG (Minimax Multi-Agent Deep Deterministic Policy Gradient)
----------------------------------------------------------------

Original Paper: https://doi.org/10.1609/aaai.v33i01.33014213

M3DDPG extends MADDPG with a minimax (robust) objective
to improve robustness against adversarial or worst-case
behaviors from other agents.

Core Idea:
- Standard multi-agent actor-critic methods assume other
  agents behave according to learned policies.
- M3DDPG instead optimizes each agent against the
  worst-case actions of opponents within a bounded set.
- This yields a robust minimax formulation.

Setting:
- Centralized training, decentralized execution.
- Each agent has:
      Deterministic actor π_i(o_i)
      Centralized critic Q_i(x, a_1, ..., a_N)

Minimax Critic Target:
For agent i:

    Q_i(x, a_i, a_-i)

Instead of directly using opponents' actions,
M3DDPG computes an adversarial perturbation:

    a_-i* = argmin_{a_-i ∈ B} Q_i(x, a_i, a_-i)

where B defines a bounded uncertainty set.

Actor Update:
- Agent i maximizes the worst-case Q-value:
      max_{a_i} min_{a_-i ∈ B} Q_i(x, a_i, a_-i)

- Implemented via gradient ascent on the
  robustified critic estimate.

Critic Update:
- Standard Bellman backup using centralized
  next-state joint actions.
- Uses deterministic policy gradient structure
  similar to MADDPG.

Rationale:
- Standard MARL can overfit to cooperative or
  predictable opponent behavior.
- Minimax training increases robustness to:
      • adversarial agents
      • modeling errors
      • non-stationary dynamics

M3DDPG = MADDPG + adversarial (minimax) opponent modeling.

It trades off optimality under nominal play
for improved robustness under worst-case interaction.
"""

import torch

from cares_reinforcement_learning.algorithm.policy import MADDPG
from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
from cares_reinforcement_learning.algorithm.configurations import M3DDPGConfig


class M3DDPG(MADDPG):
    def __init__(
        self,
        agents: list[DDPG],
        config: M3DDPGConfig,
        device: torch.device,
    ):
        super().__init__(agents, config, device)
