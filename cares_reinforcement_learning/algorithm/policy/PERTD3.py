"""
PER-TD3 (Prioritized Experience Replay for Twin Delayed Deep Deterministic Policy Gradient)
--------------------------------------------------------------

Original Paper: https://arxiv.org/abs/1511.05952

Prioritized Experience Replay (PER) modifies experience replay by sampling transitions non-uniformly
according to a priority score, typically derived from TD-error.

Priority / Sampling:
- Replay buffer stores transitions B_i and their priorities σ_i.
- Sampling probability:
      p_i = σ_i^α / sum_k σ_k^α
- Importance sampling (IS) weights correct the induced bias:
      w_i = (1 / (|B| p_i))^β
- Priorities are updated when transitions are replayed (after critic update).

How it plugs into TD3:
- TD3 remains unchanged algorithmically (actor/critic updates).
- Only the replay sampler changes:
    • sample using p_i instead of uniform
    • weight losses with w_i (commonly critic, optionally actor)
Rationale:
- Focus compute on “informative” transitions (high priority / high TD-error).
- Improve sample-efficiency vs uniform replay when priorities correlate
  with learning progress.

Notes:
- In actor-critic continuous control, PER can introduce instability and
  bias if combined with certain losses / weighting schemes.
- Careful tuning of α, β, and update rules is often required.
- Corrected variants like PAL/LAP/LA3P address these issues separately.
"""

import torch

from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.PERTD3 import Actor, Critic
from cares_reinforcement_learning.algorithm.configurations import PERTD3Config


class PERTD3(TD3):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PERTD3Config,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )
