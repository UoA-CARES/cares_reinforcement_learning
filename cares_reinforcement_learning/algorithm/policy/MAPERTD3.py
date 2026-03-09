"""
MaPER (Model-augmented Prioritized Experience Replay)
------------------------------------------------------

Original Paper: https://openreview.net/pdf?id=WuEiafqdy9H

Original Implementation: https://github.com/h-yamani/RD-PER-baselines/blob/main/MAPER/MfRL_Cont/algorithms/td3/matd3.py

MaPER extends Prioritized Experience Replay (PER) by
incorporating model-estimation errors into the priority
computation.

Core Problem:
- Standard PER prioritizes transitions using TD-error only.
- TD-error alone can be noisy due to Q-value under/overestimation.
- Early in training, Q-values are inaccurate, making TD-error
  an unreliable prioritization signal.

Core Idea:
- Augment the critic to also predict environment dynamics:
      • reward model Rθ(s, a)
      • transition model Tθ(s, a)
- Compute priority using both:
      TD-error  +  model-estimation errors

Model-Augmented Critic (MaCN):
    Cθ(s, a) = (Qθ, Rθ, Tθ)

Loss:
    LC = ξ1 LQ + ξ2 LR + ξ3 LT
    where:
        LQ = Q-value TD loss
        LR = reward prediction loss
        LT = next-state prediction loss

Priority Computation:
    σ_i = ξ1 ||δ_Q||²
        + ξ2 ||δ_R||²
        + ξ3 ||δ_T||²

Sampling probability:
    p_i ∝ σ_i^α

Key Effect:
- Early training: high model errors dominate sampling.
  → Encourages learning environment structure.
- Later training: TD-error dominates.
  → Focuses on Q-value refinement.

This induces a curriculum-like effect for critic learning.

Advantages:
- Improves sample efficiency in off-policy RL.
- Seamlessly integrates into model-free and model-based methods.
- Minimal computational overhead (shared parameters).

Scope:
- Applicable to any off-policy algorithm with Q-networks.
- Modifies critic architecture and replay priority only.
- Actor update remains unchanged.

MaPER = PER + model-error-aware prioritization
         via shared environment prediction.
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.MAPERTD3 import Actor, Critic
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.algorithm.configurations import MAPERTD3Config


class MAPERTD3(TD3):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: MAPERTD3Config,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )

        # MAPER-PER parameters
        self.scale_r = 1.0
        self.scale_s = 1.0

    def _split_output(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return target[:, 0], target[:, 1], target[:, 2:]

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state.vector_state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with fnc.evaluating(self.critic_net):
                output_one, output_two = self.critic_net(state_tensor, action_tensor)

                q_value_one, _, _ = self._split_output(output_one)
                q_value_two, _, _ = self._split_output(output_two)

                q_value = torch.minimum(q_value_one, q_value_two)

        return q_value.item()

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

        # Get current Q estimates
        output_one, output_two = self.critic_net(states, actions)
        q_values_one, predicted_rewards_one, next_states_one = self._split_output(
            output_one
        )
        q_values_two, predicted_rewards_two, next_states_two = self._split_output(
            output_two
        )

        # Difference in rewards
        diff_reward_one = 0.5 * torch.pow(
            predicted_rewards_one.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0
        ).reshape(-1, 1)

        diff_reward_two = 0.5 * torch.pow(
            predicted_rewards_two.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0
        ).reshape(-1, 1)

        # Difference in next states
        diff_next_states_one = 0.5 * torch.mean(
            torch.pow(
                next_states_one - next_states,
                2.0,
            ),
            -1,
        ).reshape(-1, 1)

        diff_next_states_two = 0.5 * torch.mean(
            torch.pow(
                next_states_two - next_states,
                2.0,
            ),
            -1,
        ).reshape(-1, 1)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_output_one, target_output_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values_one, _, _ = self._split_output(target_output_one)
            target_q_values_two, _, _ = self._split_output(target_output_two)

            target_q_values = torch.minimum(
                target_q_values_one.reshape(-1, 1), target_q_values_two.reshape(-1, 1)
            )

            predicted_rewards = (
                (
                    predicted_rewards_one.reshape(-1, 1)
                    + predicted_rewards_two.reshape(-1, 1)
                )
                / 2
            ).reshape(-1, 1)

            q_target = predicted_rewards + self.gamma * (1 - dones) * target_q_values

        diff_td_one = F.mse_loss(
            q_values_one.reshape(-1, 1), q_target, reduction="none"
        )
        diff_td_two = F.mse_loss(
            q_values_two.reshape(-1, 1), q_target, reduction="none"
        )

        critic_loss_one = (
            diff_td_one
            + self.scale_r * diff_reward_one
            + self.scale_s * diff_next_states_one
        )
        critic_loss_one = (critic_loss_one * weights.detach()).mean()

        critic_loss_two = (
            diff_td_two
            + self.scale_r * diff_reward_two
            + self.scale_s * diff_next_states_two
        )
        critic_loss_two = (critic_loss_two * weights.detach()).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        # train critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # calculate priority
        diff_td_mean = torch.cat([diff_td_one, diff_td_two], -1)
        diff_td_mean = torch.mean(diff_td_mean, 1)
        diff_td_mean = diff_td_mean.view(-1, 1)
        numpy_td_mean = diff_td_mean[:, 0].detach().data.cpu().numpy()

        diff_reward_mean = torch.cat([diff_reward_one, diff_reward_two], -1)
        diff_reward_mean = torch.mean(diff_reward_mean, 1)
        diff_reward_mean = diff_reward_mean.view(-1, 1)
        diff_reward_mean_numpy = diff_reward_mean[:, 0].detach().data.cpu().numpy()

        diff_next_state_mean = torch.cat(
            [diff_next_states_one, diff_next_states_two], -1
        )
        diff_next_state_mean = torch.mean(diff_next_state_mean, 1)
        diff_next_state_mean = diff_next_state_mean.view(-1, 1)
        diff_next_state_mean_numpy = (
            diff_next_state_mean[:, 0].detach().data.cpu().numpy()
        )

        # calculate priority
        priorities = (
            numpy_td_mean
            + self.scale_s * diff_next_state_mean_numpy
            + self.scale_r * diff_reward_mean_numpy
        )

        priorities_tensor = torch.Tensor(priorities)

        priorities = (
            priorities_tensor.clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        # Update Scales
        if self.learn_counter == 1:
            self.scale_r = np.mean(numpy_td_mean) / (
                np.mean(diff_reward_mean_numpy) + 1e-12
            )
            self.scale_s = np.mean(numpy_td_mean) / (
                np.mean(diff_next_state_mean_numpy) + 1e-12
            )

        with torch.no_grad():
            # --- TD3-style smoothing diagnostics ---
            # Noise diagnostics
            # What it tells you:
            # - target_noise_abs_mean: effective smoothing magnitude.
            # - target_noise_clip_frac high early: noise often clipped (clip too small or noise too large).
            target_noise_abs_mean = target_noise.abs().mean().item()
            target_noise_clip_frac = (
                (target_noise.abs() >= self.policy_noise_clip).float().mean().item()
            )
            info["target_noise_abs_mean"] = float(target_noise_abs_mean)
            info["target_noise_clip_frac"] = float(target_noise_clip_frac)

            # --- Component losses (unweighted, per-sample means) ---
            # These tell you whether the model heads are actually learning and on what scale.
            info["td_mse_one_mean"] = diff_td_one.mean().item()
            info["td_mse_two_mean"] = diff_td_two.mean().item()
            info["td_mse_mean"] = (
                0.5 * (diff_td_one.mean() + diff_td_two.mean())
            ).item()

            info["reward_pred_mse_one_mean"] = diff_reward_one.mean().item()
            info["reward_pred_mse_two_mean"] = diff_reward_two.mean().item()
            info["reward_pred_mse_mean"] = (
                0.5 * (diff_reward_one.mean() + diff_reward_two.mean())
            ).item()

            info["next_state_pred_mse_one_mean"] = diff_next_states_one.mean().item()
            info["next_state_pred_mse_two_mean"] = diff_next_states_two.mean().item()
            info["next_state_pred_mse_mean"] = (
                0.5 * (diff_next_states_one.mean() + diff_next_states_two.mean())
            ).item()

            # --- Scales (very important to log; they define the tradeoff) ---
            info["scale_r"] = float(self.scale_r)
            info["scale_s"] = float(self.scale_s)

            # --- Weighted contribution ratios (are aux losses dominating TD?) ---
            # These approximate how much each term contributes inside the critic loss before IS weighting.
            td_term_mean = (0.5 * (diff_td_one.mean() + diff_td_two.mean())).item()
            r_term_mean = (
                0.5 * (diff_reward_one.mean() + diff_reward_two.mean())
            ).item()
            s_term_mean = (
                0.5 * (diff_next_states_one.mean() + diff_next_states_two.mean())
            ).item()

            info["loss_term_td_mean"] = float(td_term_mean)
            info["loss_term_r_scaled_mean"] = float(self.scale_r * r_term_mean)
            info["loss_term_s_scaled_mean"] = float(self.scale_s * s_term_mean)

            den = (
                td_term_mean
                + (self.scale_r * r_term_mean)
                + (self.scale_s * s_term_mean)
                + 1e-12
            )
            info["loss_td_frac"] = float(td_term_mean / den)
            info["loss_r_frac"] = float((self.scale_r * r_term_mean) / den)
            info["loss_s_frac"] = float((self.scale_s * s_term_mean) / den)

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

            # --- Predicted reward bias (MaPER/MAPERTD3-specific) ---
            # Since the Bellman target uses predicted reward instead of env reward,
            # any systematic bias here directly shifts Q-targets and can cause
            # value inflation or suppression.
            predicted_reward_mean = predicted_rewards.mean().item()
            env_reward_mean = rewards.mean().item()

            info["predicted_reward_mean"] = predicted_reward_mean
            info["env_reward_mean"] = env_reward_mean
            info["predicted_reward_bias"] = predicted_reward_mean - env_reward_mean

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

            # ---Priority diagnostics (raw + final PER priorities) ---
            prio_td = diff_td_mean.squeeze(1)  # (B,)
            prio_r = self.scale_r * diff_reward_mean.squeeze(1)  # (B,)
            prio_s = self.scale_s * diff_next_state_mean.squeeze(1)  # (B,)
            prio_raw = prio_td + prio_r + prio_s  # (B,)

            info["priority_raw_mean"] = prio_raw.mean().item()
            info["priority_raw_p95"] = prio_raw.quantile(0.95).item()

            prio_den = prio_raw.mean().item() + 1e-12
            info["priority_td_frac"] = float(prio_td.mean().item() / prio_den)
            info["priority_r_frac"] = float(prio_r.mean().item() / prio_den)
            info["priority_s_frac"] = float(prio_s.mean().item() / prio_den)

            prio_post = prio_raw.clamp(min=self.min_priority).pow(self.per_alpha)
            info["priority_mean"] = prio_post.mean().item()
            info["priority_p95"] = prio_post.quantile(0.95).item()
            info["priority_max"] = prio_post.max().item()

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities

    def _update_actor(
        self, states: torch.Tensor, weights: torch.Tensor
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        actions = self.actor_net(states.detach())

        with fnc.evaluating(self.critic_net):
            output_one, output_two = self.critic_net(states.detach(), actions)

        actor_q_one, _, _ = self._split_output(output_one)
        actor_q_two, _, _ = self._split_output(output_two)

        actor_q_values = torch.minimum(actor_q_one, actor_q_two)

        actor_loss = -(actor_q_values * weights).mean()

        # ---------------------------------------------------------
        # Deterministic Policy Gradient Strength (∇a Q(s,a))
        # ---------------------------------------------------------
        # Measures how steep the critic surface is w.r.t. actions.
        # ~0 early  -> critic flat, actor receives no learning signal.
        # Very large -> critic overly sharp, can cause unstable actor updates.
        dq_da = torch.autograd.grad(
            outputs=actor_loss,
            inputs=actions,
            retain_graph=True,  # because we do backward(actor_loss) next
            create_graph=False,  # diagnostic only
            allow_unused=False,
        )[0]
        with torch.no_grad():
            # - ~0 early: critic surface flat around actor actions (weak learning signal)
            # - very large: critic surface sharp -> unstable / exploitative actor updates
            info["dq_da_abs_mean"] = dq_da.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da.norm(dim=1).quantile(0.95).item()

        # Optimize the actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        with torch.no_grad():
            # Policy Action Health (tanh policies in [-1, 1])
            # pi_action_saturation_frac:
            # High values (>0.8 early) often mean the actor is slamming bounds,
            # reducing effective gradient flow through tanh.
            info["pi_action_mean"] = actions.mean().item()
            info["pi_action_std"] = actions.std().item()
            info["pi_action_abs_mean"] = actions.abs().mean().item()
            info["pi_action_saturation_frac"] = (
                (actions.abs() > 0.95).float().mean().item()
            )

            # actor_q_mean should generally increase over training.
            # actor_q_std large + unstable may indicate critic inconsistency.
            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_values.mean().item()
            info["actor_q_std"] = actor_q_values.std().item()
        return info
