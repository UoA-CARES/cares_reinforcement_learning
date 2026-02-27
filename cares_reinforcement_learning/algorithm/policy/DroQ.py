"""
DroQ (Dropout Q-Functions for Continuous Control)
---------------------------------------------------

Original Paper: https://openreview.net/pdf?id=xCVJMsPv3RT

DroQ is an off-policy actor-critic method that improves
sample efficiency by combining high update-to-data ratios
with dropout-regularized Q-networks.

Core Problem:
- Increasing the update-to-data ratio (UTD) improves
  sample efficiency (as in REDQ).
- However, many critic updates per environment step
  can cause overfitting to replay data.
- Overfitting leads to value overestimation and instability.

Core Idea:
- Apply dropout to Q-networks during training.
- Keep the architecture simple (no large ensembles).
- Use high UTD (many gradient updates per step).
- Maintain clipped double Q-learning (as in SAC).

Architecture:
- Stochastic Gaussian actor (SAC-style).
- Twin Q-networks with dropout layers.
- Target networks for stable bootstrapping.

Critic Target:
    a' ~ π(s')
    y = r + γ ( min(Q1', Q2') - α log π(a'|s') )

Critic Loss:
    MSE between Q(s,a) and y,
    with dropout active during training.

Actor Update:
    Standard SAC objective:
        maximize E[ min(Q1, Q2) - α log π ]

Key Behaviour:
- Dropout acts as regularization against replay overfitting.
- Enables high UTD without instability.
- No need for large Q-ensembles (unlike REDQ).

Advantages:
- Strong sample efficiency.
- Minimal architectural complexity.
- Lower computational cost than large ensembles.
- Easy extension of SAC.

DroQ = SAC + high update-to-data ratio
       + dropout-regularized critics.
"""

import torch

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.DroQ import Actor, Critic
from cares_reinforcement_learning.algorithm.configurations import DroQConfig


class DroQ(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: DroQConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)
