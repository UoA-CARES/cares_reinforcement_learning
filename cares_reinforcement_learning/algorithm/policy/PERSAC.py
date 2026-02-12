"""
PER-SAC (Prioritized Experience Replay for Soft Actor-Critic)
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

How it plugs into SAC:
- SAC remains unchanged algorithmically (actor/critic/entropy updates).
- Only the replay sampler changes:
    • sample using p_i instead of uniform
    • weight losses with w_i (commonly critic, optionally actor/alpha)

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

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.PERSAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import PERSACConfig


class PERSAC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PERSACConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)
