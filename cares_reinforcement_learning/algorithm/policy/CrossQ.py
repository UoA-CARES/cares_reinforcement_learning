"""
CrossQ (Cross-Normalized Q-Learning)
-------------------------------------

Original Paper: https://arxiv.org/pdf/1902.05605
Code based on: https://github.com/modelbased/minirllab/blob/main/agents/sac_crossq.py

CrossQ is an off-policy actor-critic method that improves
sample efficiency by stabilizing critic training through
cross-normalization of features across current and target
Q-networks.

Core Problem:
- SAC / TD3 often avoid BatchNorm in critics because
  target networks and bootstrapping make statistics unstable.
- Poor feature scaling limits update-to-data ratio (UTD).
- High UTD typically requires ensembles (e.g., REDQ).

Core Idea:
- Use Batch Normalization in Q-networks.
- Compute normalization statistics jointly across:
      • current Q forward pass
      • target Q forward pass
- This “cross” normalization aligns feature distributions
  and stabilizes bootstrapped targets.

Critic Target (SAC-style):
    a' ~ π(s')
    y = r + γ ( min(Q1', Q2') - α log π(a'|s') )

But during forward passes:
    - Concatenate inputs for Q and Q'
    - Apply shared BatchNorm statistics
    - Split outputs afterward

This removes the distribution mismatch that normally
destabilizes BatchNorm in RL.

Actor Update:
    Same as SAC:
        maximize E[ min(Q1, Q2) - α log π ]

Key Behaviour:
- Enables high update-to-data ratios without ensembles.
- Improves sample efficiency.
- Requires only twin critics (no large ensembles).
- Minimal architectural overhead.

Advantages:
- Strong performance with simple modifications.
- No need for REDQ-style large critic sets.
- Maintains SAC stability while improving data usage.

CrossQ = SAC + cross-normalized critics
         enabling stable high-UTD training.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.CrossQ import Actor, Critic
from cares_reinforcement_learning.algorithm.configurations import CrossQConfig


class CrossQ(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: CrossQConfig,
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

        cat_states = torch.cat([states, next_states], dim=0)
        cat_actions = torch.cat([actions, next_actions], dim=0)

        cat_q_values_one, cat_q_values_two = self.critic_net(cat_states, cat_actions)

        q_values_one, q_values_one_next = torch.chunk(cat_q_values_one, chunks=2, dim=0)
        q_values_two, q_values_two_next = torch.chunk(cat_q_values_two, chunks=2, dim=0)

        with torch.no_grad():
            min_next_q = torch.minimum(q_values_one_next, q_values_two_next)
            next_q_values = min_next_q - self.alpha * next_log_pi

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * next_q_values
            )

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
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

            # ---------------------------------------------------------
            # CrossQ-specific diagnostics
            # ---------------------------------------------------------
            # (1) Self-bootstrap "optimism": if q_next is systematically larger than q_now
            # it can indicate overestimation pressure since the same network supplies bootstrap values.
            info["q_next_minus_q_mean"] = (
                (min_next_q - torch.minimum(q_values_one, q_values_two)).mean().item()
            )

            # (2) Next-Q magnitude vs current-Q magnitude (scale drift check)
            info["q_next_abs_mean"] = min_next_q.abs().mean().item()
            info["q_abs_mean"] = (
                torch.minimum(q_values_one, q_values_two).abs().mean().item()
            )

            # (3) CrossQ concatenation health: are the two halves numerically similar in distribution?
            # Big discrepancies can indicate distribution shift or implementation bugs in cat/chunk wiring.
            info["crossq_half_gap_abs_mean"] = (
                (torch.minimum(q_values_one, q_values_two).mean() - min_next_q.mean())
                .abs()
                .item()
            )

            # ---------------------------------------------------------
            # CrossQ "bootstrap-from-self" next critics (s',a')
            # (these are NOT target critics; they're the next-half outputs)
            # ---------------------------------------------------------
            info["q1_next_mean"] = q_values_one_next.mean().item()
            info["q2_next_mean"] = q_values_two_next.mean().item()
            info["q_next_twin_gap_abs_mean"] = (
                (q_values_one_next - q_values_two_next).abs().mean().item()
            )

            # ---------------------------------------------------------
            # Soft target decomposition (same as SAC, but using q_next from critic_net)
            # ---------------------------------------------------------
            # alpha_log_pi is typically negative; entropy_bonus is typically positive
            alpha_log_pi = self.alpha * next_log_pi
            # this is what gets ADDED to minQ in the target
            entropy_bonus = -self.alpha * next_log_pi

            soft_target_value = min_next_q + entropy_bonus  # == minQ - alpha*log_pi

            info["next_min_q_mean"] = min_next_q.mean().item()
            info["alpha_log_pi_mean"] = alpha_log_pi.mean().item()
            info["entropy_bonus_mean"] = entropy_bonus.mean().item()
            info["soft_target_value_mean"] = soft_target_value.mean().item()

            # --- Bellman target scale (reward scaling / discount sanity) ---
            # If q_target drifts upward without reward improvement, suspect reward_scale, gamma, or instability.
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # --- TD error diagnostics (Bellman fit quality) ---
            # td_abs_mean down over time is healthy; persistent growth/spikes often indicate critic instability.
            td1 = q_values_one - q_target  # signed
            td2 = q_values_two - q_target  # signed

            info["td1_mean"] = td1.mean().item()
            info["td1_std"] = td1.std(unbiased=False).item()
            info["td1_abs_mean"] = td1.abs().mean().item()

            info["td2_mean"] = td2.mean().item()
            info["td2_std"] = td2.std(unbiased=False).item()
            info["td2_abs_mean"] = td2.abs().mean().item()

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities
