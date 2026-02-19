"""
TQC (Truncated Quantile Critics)
---------------------------------

Original Paper: https://arxiv.org/abs/1812.05905
Original Code: https://github.com/SamsungLabs/tqc_pytorch

TQC extends Soft Actor-Critic (SAC) using distributional
critics and quantile truncation to reduce overestimation bias.

Core Problem:
- Clipped double Q (min of two critics) reduces
  overestimation but may still be insufficient.
- Distributional RL provides richer return estimates,
  but naive use can still overestimate.

Core Idea:
- Use N distributional critics.
- Each critic outputs K quantiles of the return distribution.
- When forming the target, drop the top quantiles
  (highest return estimates).
- This truncation makes the target more conservative.

Critic Output:
    Z_i(s,a) = { q_i^1, q_i^2, ..., q_i^K }

Target Construction:
    - Sample next action from current policy.
    - Collect all quantiles from all target critics.
    - Sort quantiles.
    - Remove the highest M quantiles.
    - Use remaining quantiles to compute target.

Critic Loss:
    - Quantile regression loss (Huber quantile loss)
      between predicted and truncated target quantiles.

Actor Update:
    - Same structure as SAC:
        maximize expected Q under policy
    - Uses mean of quantile estimates.

Key Behaviour:
- Truncation reduces optimistic bias.
- Larger ensembles increase stability.
- Maintains entropy-regularized SAC objective.

Advantages:
- Stronger bias control than clipped double Q.
- High sample efficiency.
- Minimal change to SAC structure.

TQC = SAC + distributional critics +
      quantile truncation for conservative targets.
"""

from typing import Any

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.TQC import Actor, Critic
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import TQCConfig
from cares_reinforcement_learning.types.observation import SARLObservation


class TQC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: TQCConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )

        # TQC specific parameters
        self.num_quantiles = config.num_quantiles
        self.num_critics = config.num_critics
        self.kappa = config.kappa

        self.quantiles_total = self.num_quantiles * self.num_critics

        self.top_quantiles_to_drop = config.top_quantiles_to_drop

        self.quantile_taus = torch.FloatTensor(
            [
                i / self.num_quantiles + 0.5 / self.num_quantiles
                for i in range(0, self.num_quantiles)
            ]
        ).to(device)

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state.vector_state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                q_value = (
                    self.critic_net(state_tensor, action_tensor)
                    .mean(2)
                    .mean(1, keepdim=True)
                )

        return q_value.item()

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> tuple[dict[str, Any], np.ndarray]:
        info: dict[str, Any] = {}
        batch_size = len(states)

        drop_q_range = self.quantiles_total - self.top_quantiles_to_drop
        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            # compute and cut quantiles at the next state
            # batch x nets x quantiles
            target_q_values = self.target_critic_net(next_states, next_actions)
            sorted_target_q_values, _ = torch.sort(
                target_q_values.reshape(batch_size, -1)
            )
            top_quantile_target_q_values = sorted_target_q_values[:, :drop_q_range]

            # compute target
            q_target = rewards + (1 - dones) * self.gamma * (
                top_quantile_target_q_values - self.alpha * next_log_pi
            )

        q_values = self.critic_net(states, actions)

        # Compute td_error for PER
        sorted_q_values, _ = torch.sort(q_values.reshape(batch_size, -1))
        top_quantile_q_values = sorted_q_values[:, :drop_q_range]

        td_errors = top_quantile_q_values - q_target
        td_errors_abs = td_errors.abs()
        td_error = td_errors_abs.mean(dim=1)  # mean over quantiles

        critic_loss_total = hlp.calculate_quantile_huber_loss(
            q_values,
            q_target,
            self.quantile_taus,
            kappa=self.kappa,
            use_pairwise_loss=True,
            use_mean_reduction=True,
            use_quadratic_smoothing=True,
        )

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            td_error.clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        with torch.no_grad():
            # ---- Loss ----
            info["critic_loss_total"] = critic_loss_total.item()

            # ---- Target decomposition (SAC-like, but on trimmed quantiles) ----
            # Conservative bootstrap term is the mean of kept target quantiles (pre-entropy)
            info["target_min_q_mean"] = top_quantile_target_q_values.mean().item()

            # --- Soft target decomposition (SAC-specific) ---
            # min_target_q_mean: the conservative bootstrap value from twin critics (pre-entropy)
            # entropy_term_mean: magnitude of entropy regularization in the target (alpha * log_pi is usually negative)
            # soft_target_value_mean: the exact term used inside the Bellman target before reward/discount
            # alpha_log_pi is typically negative; entropy_bonus is typically positive
            alpha_log_pi = self.alpha * next_log_pi
            # this is what gets ADDED to minQ in the target
            entropy_bonus = -self.alpha * next_log_pi

            soft_target_value = top_quantile_target_q_values + entropy_bonus

            info["alpha_log_pi_mean"] = alpha_log_pi.mean().item()
            info["entropy_bonus_mean"] = entropy_bonus.mean().item()
            info["soft_target_value_mean"] = soft_target_value.mean().item()

            # ---- Bellman target scale ----
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # ---- Current quantiles scale ----
            q_flat = q_values.reshape(batch_size, -1)  # (B, Q_total)
            info["q_mean"] = q_flat.mean().item()
            info["q_std"] = q_flat.std(unbiased=False).item()

            # ---- Quantile trimming diagnostics (TQC-specific) ----
            # If dropped quantiles are very far above kept ones, trimming is doing real work
            # - kept_mean: mean of the quantiles we *keep* (conservative estimate used for learning)
            # - dropped_mean: mean of the high quantiles we *drop* (optimistic tail)
            # - drop_gap_mean: how far the dropped tail sits above the kept mass
            #   Large gap => trimming is actively removing optimistic tail pressure.
            info["tqc_kept_mean"] = top_quantile_q_values.mean().item()
            dropped_q_values = sorted_q_values[:, drop_q_range:]
            kept_q_mean_per_sample = top_quantile_q_values.mean(dim=1)
            dropped_q_mean_per_sample = dropped_q_values.mean(dim=1)

            info["tqc_dropped_mean"] = dropped_q_values.mean().item()
            info["tqc_drop_gap_mean"] = (
                (dropped_q_mean_per_sample - kept_q_mean_per_sample).mean().item()
            )

            # Same idea on target side
            info["tqc_target_kept_mean"] = top_quantile_target_q_values.mean().item()
            dropped_target_q_values = sorted_target_q_values[:, drop_q_range:]
            kept_target_mean_per_sample = top_quantile_target_q_values.mean(dim=1)
            dropped_target_mean_per_sample = dropped_target_q_values.mean(dim=1)

            info["tqc_target_dropped_mean"] = dropped_target_q_values.mean().item()
            info["tqc_target_drop_gap_mean"] = (
                (dropped_target_mean_per_sample - kept_target_mean_per_sample)
                .mean()
                .item()
            )

            # ---- Quantile spread (uncertainty/sharpness proxies) ----
            # IQR across kept quantiles: (q75 - q25) per sample
            q25 = top_quantile_q_values.quantile(0.25, dim=1)
            q50 = top_quantile_q_values.quantile(0.50, dim=1)
            q75 = top_quantile_q_values.quantile(0.75, dim=1)
            iqr = q75 - q25

            info["q_iqr_mean"] = iqr.mean().item()
            info["q_iqr_p95"] = iqr.quantile(0.95).item()
            info["q_median_mean"] = q50.mean().item()

            # Same IQR on target kept quantiles
            tq25 = top_quantile_target_q_values.quantile(0.25, dim=1)
            tq50 = top_quantile_target_q_values.quantile(0.50, dim=1)
            tq75 = top_quantile_target_q_values.quantile(0.75, dim=1)
            tiqr = tq75 - tq25

            info["target_q_iqr_mean"] = tiqr.mean().item()
            info["target_q_iqr_p95"] = tiqr.quantile(0.95).item()
            info["target_q_median_mean"] = tq50.mean().item()

            # ---- TD-error diagnostics (fit quality + tails) ----
            info["td_abs_mean"] = td_error.mean().item()
            info["td_abs_std"] = td_error.std(unbiased=False).item()
            info["td_abs_p95"] = td_error.quantile(0.95).item()
            info["td_abs_max"] = td_error.max().item()

            # Quantile-level TD tail (more sensitive than mean-over-quantiles)
            # Per sample: max |TD| among kept quantiles
            td_abs_qmax = td_errors_abs.max(dim=1).values
            info["td_abs_qmax_mean"] = td_abs_qmax.mean().item()
            info["td_abs_qmax_p95"] = td_abs_qmax.quantile(0.95).item()
            info["td_abs_qmax_max"] = td_abs_qmax.max().item()

        return info, priorities

    def _update_actor_alpha(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        pi, log_pi, _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            q_quant = self.critic_net(states, pi)

            q_mean_per_critic = q_quant.mean(dim=2)  # (B, C)
            mean_qf_pi = q_mean_per_critic.mean(dim=1, keepdim=True)  # (B, 1)

        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()

        # ---------------------------------------------------------
        # Stochastic Policy Gradient Strength (∇a [α log π(a|s) − Q(s,a)])
        # ---------------------------------------------------------
        # Measures how steep the entropy-regularized critic objective is
        # w.r.t. the sampled policy actions.
        #
        # ~0 early  -> critic surface and entropy term nearly flat;
        #              actor receives weak learning signal.
        #
        # Very large -> critic or entropy term is very sharp around policy
        #               actions; can lead to unstable or overly aggressive
        #               actor updates.
        dq_da = torch.autograd.grad(
            outputs=actor_loss,
            inputs=pi,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        with torch.no_grad():
            info["dq_da_abs_mean"] = dq_da.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da.norm(dim=1).quantile(0.95).item()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        with torch.no_grad():
            # --- Policy entropy diagnostics ---
            info["log_pi_mean"] = log_pi.mean().item()
            info["log_pi_std"] = log_pi.std(unbiased=False).item()

            # --- Action magnitude/saturation ---
            info["pi_action_abs_mean"] = pi.abs().mean().item()
            info["pi_action_std"] = pi.std(unbiased=False).item()
            info["pi_action_saturation_frac"] = (pi.abs() > 0.95).float().mean().item()

            # --- On-policy critic signal (TQC uses mean over critics+quantiles) ---
            info["mean_qf_pi_mean"] = mean_qf_pi.mean().item()
            info["mean_qf_pi_std"] = mean_qf_pi.std(unbiased=False).item()

            # --- Critic disagreement at policy actions (TQC analogue of twin-gap) ---
            # Useful dispersion metrics (TQC analogue of "twin gap")
            q_std_across_critics = q_mean_per_critic.std(dim=1, unbiased=False)  # (B,)
            q_range_across_critics = (
                q_mean_per_critic.max(dim=1).values
                - q_mean_per_critic.min(dim=1).values
            )  # (B,)

            # Quantile spread (distributional uncertainty / sharpness)
            # IQR across quantiles after averaging critics
            q_mean_across_critics = q_quant.mean(dim=1)  # (B, N)
            q_q25 = q_mean_across_critics.quantile(0.25, dim=1)  # (B,)
            q_q50 = q_mean_across_critics.quantile(0.50, dim=1)  # (B,)
            q_q75 = q_mean_across_critics.quantile(0.75, dim=1)  # (B,)
            q_iqr = q_q75 - q_q25  # (B,)

            info["q_pi_critics_std_mean"] = q_std_across_critics.mean().item()
            info["q_pi_critics_std_p95"] = q_std_across_critics.quantile(0.95).item()
            info["q_pi_critics_range_mean"] = q_range_across_critics.mean().item()
            info["q_pi_critics_range_p95"] = q_range_across_critics.quantile(
                0.95
            ).item()

            # --- Quantile spread (distributional sharpness / uncertainty proxy) ---
            info["q_pi_quantile_iqr_mean"] = q_iqr.mean().item()
            info["q_pi_quantile_iqr_p95"] = q_iqr.quantile(0.95).item()
            info["q_pi_quantile_median_mean"] = q_q50.mean().item()

            # --- Entropy gap (alpha tuning health) ---
            entropy_gap = -(log_pi + self.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()
            info["entropy_gap_std"] = entropy_gap.std(unbiased=False).item()

            # --- Losses / temperature ---
            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()
            info["log_alpha"] = self.log_alpha.item()

        return info
