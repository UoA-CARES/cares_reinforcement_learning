"""
LAP (Loss-Adjusted Prioritized Experience Replay)
--------------------------------------------------

Original Paper: https://arxiv.org/abs/2007.06049

LAP modifies Prioritized Experience Replay (PER) to remove
bias introduced by combining TD-error-based sampling with
mean-squared error (MSE) critic losses.

Core Problem:
- PER samples transitions with probability ∝ |TD error|^α.
- When combined with MSE, this induces a biased objective.
- High-error outliers can dominate updates and destabilize
  off-policy learning, especially in continuous control.

Core Idea:
- Adjust the critic loss so that prioritized sampling and
  uniform sampling produce the same expected gradient.
- Prevent outlier bias leakage while preserving PER's
  sample-efficiency benefits.

Key Modifications:

1) Clipped Priorities
   p_i = max(|δ_i|^α, 1)
   - Prevents very small priorities.
   - Removes need for ε-offset.

2) Huber Loss (κ = 1)
   - Replaces MSE in critic updates.
   - Behaves like MSE near zero.
   - Behaves like L1 for large TD errors.
   - Limits influence of outliers.

Result:
- Prioritized sampling remains proportional to TD error.
- Critic updates are no longer biased by the interaction
  between MSE and non-uniform sampling.
- Empirically improves stability over vanilla PER in
  off-policy actor-critic methods.

Scope:
- Applies to any off-policy algorithm using PER
  (e.g., DQN, TD3, SAC).
- Only modifies critic loss and priority calculation.
- Actor updates remain unchanged.

LAP = PER + Huber critic loss + clipped priorities.
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.algorithm.lossess as loss
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.configurations import LAPSACConfig
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.LAPSAC import Actor, Critic


class LAPSAC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: LAPSACConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )

            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                self.reward_scale * rewards + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        huber_lose_one = loss.calculate_huber_loss(
            td_error_one, self.min_priority, use_quadratic_smoothing=False
        )
        huber_lose_two = loss.calculate_huber_loss(
            td_error_two, self.min_priority, use_quadratic_smoothing=False
        )
        critic_loss_total = huber_lose_one + huber_lose_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        with torch.no_grad():
            # --- Twin critic disagreement (stability/uncertainty) ---
            # If this grows over training, critics are diverging / becoming inconsistent.
            info["q1_mean"] = q_values_one.mean().item()
            info["q2_mean"] = q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (q_values_one - q_values_two).abs().mean().item()
            )

            # --- Target critics disagreement (target stability) ---
            # Large/unstable gap here often means target critics are drifting or policy is visiting OOD actions.
            info["target_q1_mean"] = target_q_values_one.mean().item()
            info["target_q2_mean"] = target_q_values_two.mean().item()
            info["target_q_twin_gap_abs_mean"] = (
                (target_q_values_one - target_q_values_two).abs().mean().item()
            )

            # --- Soft target decomposition (SAC-specific) ---
            # min_target_q_mean: the conservative bootstrap value from twin critics (pre-entropy)
            # entropy_term_mean: magnitude of entropy regularization in the target (alpha * log_pi is usually negative)
            # soft_target_value_mean: the exact term used inside the Bellman target before reward/discount
            min_target_q = torch.minimum(target_q_values_one, target_q_values_two)
            entropy_term = self.alpha * next_log_pi  # typically negative
            soft_target_value = min_target_q - entropy_term  # == minQ - alpha*log_pi

            info["target_min_q_mean"] = min_target_q.mean().item()
            info["entropy_term_mean"] = entropy_term.mean().item()
            info["soft_target_value_mean"] = soft_target_value.mean().item()

            # --- Bellman target scale (reward scaling / discount sanity) ---
            # If q_target drifts upward without reward improvement, suspect reward_scale, gamma, or instability.
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std().item()

            # --- TD error diagnostics (Bellman fit quality) ---
            # td_abs_mean down over time is healthy; persistent growth/spikes often indicate critic instability.
            td1 = q_values_one - q_target  # signed
            td2 = q_values_two - q_target  # signed

            info["td1_mean"] = td1.mean().item()
            info["td1_std"] = td1.std().item()
            info["td1_abs_mean"] = td1.abs().mean().item()

            info["td2_mean"] = td2.mean().item()
            info["td2_std"] = td2.std().item()
            info["td2_abs_mean"] = td2.abs().mean().item()

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["critic_loss_one"] = huber_lose_one.item()
            info["critic_loss_two"] = huber_lose_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities
