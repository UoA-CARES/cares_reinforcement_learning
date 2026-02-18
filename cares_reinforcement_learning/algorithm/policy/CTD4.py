"""
CTD4 (Continuous Twin Delayed Distributional Deterministic Policy Gradient)
----------------------------------------------------------------------------

Original Paper: https://arxiv.org/abs/2405.02576

Original Code: https://github.com/UoA-CARES/cares_reinforcement_learning/blob/1fce6fcde5183bafe4efce0aa30fc59f630a8429/cares_reinforcement_learning/algorithm/policy/CTD4.py

This algorithm extends TD3 by replacing scalar Q-value critics
with continuous distributional critics. Each critic outputs a
Gaussian return distribution parameterized by (μ, σ).

Data / Training (off-policy):
- Uses standard TD3 replay buffer and target networks.
- Each critic predicts Z(s, a) ~ Normal(μ, σ).
- Actor remains deterministic.

Distributional Bellman update:
- Target critics produce multiple distributions for next state.
- Critic outputs are fused (Kalman / average / minimum).
- Target distribution is constructed analytically:
      μ_target  = r + γ μ_fused (1 - done)
      σ_target  = γ σ_fused
- No categorical projection step is required.

Critic updates:
- Each critic minimizes KL divergence between:
      Z_current  and  Z_target
- Critics are optimized independently (ensemble).

Actor updates:
- Actor maximizes fused mean return:
      J ≈ E[ μ_fused(s, π(s)) ]
- Implemented as minimizing -μ_fused.mean().

Ensemble fusion:
- Default: Kalman fusion (uncertainty-weighted Gaussian fusion).
- Alternatives: average or minimum.
- Kalman fusion mitigates overestimation without discarding
  ensemble information.

Rationale:
- Continuous distributions avoid categorical support tuning
  and projection steps.
- KL between Gaussians is analytic and stable.
- Ensemble fusion reduces overestimation bias while preserving
  uncertainty information.

CTD4 = TD3 + Gaussian distributional critics + Kalman fusion.
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.CTD4 import Actor, Critic
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import CTD4Config


class CTD4(TD3):
    critic_net: Critic
    target_critic_net: Critic

    def __init__(
        self,
        actor_network: Actor,
        ensemble_critic: Critic,
        config: CTD4Config,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=ensemble_critic,
            config=config,
            device=device,
        )

        self.fusion_method = config.fusion_method
        self.kalman_alpha = config.kalman_alpha

        self.lr_ensemble_critic = config.critic_lr
        self.ensemble_critic_optimizers = [
            torch.optim.Adam(
                critic_net.parameters(),
                lr=self.lr_ensemble_critic,
                **config.critic_lr_params,
            )
            for critic_net in self.critic_net.critics
        ]

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.tensor(
            state.vector_state, dtype=torch.float32, device=self.device
        )
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.unsqueeze(0)

        q_u_set = []
        q_std_set = []

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                for critic_net in self.critic_net.critics:
                    actor_q_u, actor_q_std = critic_net(state_tensor, action_tensor)

                    q_u_set.append(actor_q_u)
                    q_std_set.append(actor_q_std)

        fusion_u_a, _, _ = self._fuse_critic_outputs(1, q_u_set, q_std_set)

        return fusion_u_a.item()

    def _kalman_interpolated(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        num_critics = len(u_set)
        # Start from critic 0
        fusion_u = u_set[0]  # (B,1)
        fusion_std = std_set[0]  # (B,1)

        # weights: (B,E), start fully on critic 0
        weights = torch.zeros(
            (batch_size, num_critics), device=self.device, dtype=torch.float32
        )
        weights[:, 0] = 1.0

        # Fuse critics 1..E-1 sequentially
        for i in range(1, num_critics):
            x2 = u_set[i]  # (B,1)
            std2 = std_set[i]  # (B,1)

            # Kalman gain: trust weight on the NEW critic (x2) relative to current fused estimate
            # K close to 1 -> new critic dominates; K close to 0 -> old fused dominates
            kalman_gain = (fusion_std**2) / (fusion_std**2 + std2**2 + 1e-12)  # (B,1)

            # Mean fusion: fused <- (1-K) * fused + K * x2
            fusion_u = fusion_u + kalman_gain * (x2 - fusion_u)

            # Variance fusion (your "interpolated" rule): var <- (1-K)*var1 + K*var2
            fusion_variance = (
                (1 - kalman_gain) * (fusion_std**2) + kalman_gain * (std2**2) + 1e-6
            )
            fusion_std = torch.sqrt(fusion_variance)

            # Weight update:
            # - all existing contributions get down-weighted by (1-K)
            # - new critic i gets weight K
            weights = weights * (1 - kalman_gain)  # broadcast (B,E) * (B,1)
            weights[:, i : i + 1] = (
                weights[:, i : i + 1] + kalman_gain
            )  # add (B,1) into column i

        return fusion_u, fusion_std, weights

    def _kalman_precision(self, u_set, std_set):
        u_mat = torch.concat(u_set, dim=1)  # (B,E)
        std_mat = torch.concat(std_set, dim=1)  # (B,E)

        eps = 1e-12
        precision = 1.0 / (std_mat**2 + eps)
        precision_sum = precision.sum(dim=1, keepdim=True)

        weights = precision / precision_sum
        fusion_u = (weights * u_mat).sum(dim=1, keepdim=True)

        fusion_var = self.kalman_alpha / (precision_sum + eps)
        fusion_std = torch.sqrt(fusion_var + 1e-6)

        return fusion_u, fusion_std, weights

    def _average(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Average value among the critic predictions:
        u_mat = torch.concat(u_set, dim=1)  # (B,E)
        std_mat = torch.concat(std_set, dim=1)  # (B,E)

        fusion_u = u_mat.mean(dim=1, keepdim=True)  # (B,1)
        fusion_std = std_mat.mean(dim=1, keepdim=True)  # (B,1)

        E = u_mat.shape[1]
        weights = torch.full((batch_size, E), 1.0 / E, device=u_mat.device)

        return fusion_u, fusion_std, weights

    def _minimum(
        self, u_set: list[torch.Tensor], std_set: list[torch.Tensor], batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        u_mat = torch.concat(u_set, dim=1)  # (B,E)
        std_mat = torch.concat(std_set, dim=1)  # (B,E)

        min_vals, min_idx = torch.min(u_mat, dim=1)  # (B,), (B,)
        fusion_u = min_vals.unsqueeze(1)  # (B,1)

        # std of the selected critic
        fusion_std = std_mat[torch.arange(batch_size), min_idx].unsqueeze(1)  # (B,1)

        # one-hot weights
        E = u_mat.shape[1]
        weights = torch.zeros((batch_size, E), device=u_mat.device)
        weights[torch.arange(batch_size), min_idx] = 1.0

        return fusion_u, fusion_std, weights

    def _fuse_critic_outputs(
        self, batch_size: int, u_set: list[torch.Tensor], std_set: list[torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.fusion_method == "kalman_precision":
            fusion_u, fusion_std, weights = self._kalman_precision(u_set, std_set)
        elif self.fusion_method == "kalman_interpolated":
            fusion_u, fusion_std, weights = self._kalman_interpolated(
                u_set, std_set, batch_size
            )
        elif self.fusion_method == "average":
            fusion_u, fusion_std, weights = self._average(u_set, std_set, batch_size)
        elif self.fusion_method == "minimum":
            fusion_u, fusion_std, weights = self._minimum(u_set, std_set, batch_size)
        else:
            raise ValueError(
                f"Invalid fusion method: {self.fusion_method}. Please choose between 'kalman_precision', 'kalman_interpolated', 'average', or 'minimum'."
            )

        return fusion_u, fusion_std, weights

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

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)

            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )

            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            u_set = []
            std_set = []

            for target_critic_net in self.target_critic_net.critics:
                u, std = target_critic_net(next_states, next_actions)

                u_set.append(u)
                std_set.append(std)

            fusion_u, fusion_std, fusion_weights = self._fuse_critic_outputs(
                batch_size, u_set, std_set
            )

            # Create the target distribution = aX+b
            u_target = rewards + self.gamma * fusion_u * (1 - dones)
            std_target = self.gamma * fusion_std

            target_distribution = torch.distributions.normal.Normal(
                u_target, std_target
            )

        critic_loss_totals = []
        critic_loss_elementwise = []

        # --- current (s,a) ensemble output health (optional but useful) ---
        current_mu_means: list[float] = []
        current_sigma_means: list[float] = []

        for critic_net, critic_net_optimiser in zip(
            self.critic_net.critics, self.ensemble_critic_optimizers
        ):
            u_current, std_current = critic_net(states, actions)
            current_distribution = torch.distributions.normal.Normal(
                u_current, std_current
            )

            # Compute each critic loss as KL divergence to the target distribution
            critic_elementwise_loss = torch.distributions.kl.kl_divergence(
                current_distribution, target_distribution
            )
            critic_loss_elementwise.append(critic_elementwise_loss)

            critic_loss = critic_elementwise_loss.mean()
            critic_loss_totals.append(critic_loss.item())

            # If σ collapses while KL stays high -> overconfident wrong critic (bad calibration)
            current_mu_means.append(u_current.mean().item())
            current_sigma_means.append(std_current.mean().item())

            critic_net_optimiser.zero_grad()
            critic_loss.backward()
            critic_net_optimiser.step()

        kl_stack = torch.stack(critic_loss_elementwise, dim=0)
        critic_max_per_sample = torch.max(kl_stack, dim=0).values

        # Update the Priorities - PER only
        priorities = (
            critic_max_per_sample.clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
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

            # How different are the critics’ average predicted means from each other (on the current batch)?
            info["mu_std_across_critics"] = float(np.std(current_mu_means))
            info["sigma_std_across_critics"] = float(np.std(current_sigma_means))

            # --- Target ensemble diagnostics (s', a') ---
            u_mat = torch.concat(u_set, dim=1)  # (B, E)
            std_mat = torch.concat(std_set, dim=1)  # (B, E)

            # mu_std_mean is “ensemble disagreement” (epistemic spread). Spikes/growth often = divergence/OOD.
            info["target_ensemble_mu_mean"] = u_mat.mean().item()
            info["target_ensemble_mu_std_mean"] = (
                u_mat.std(dim=1, unbiased=False).mean().item()
            )

            # sigma_mean is average predicted uncertainty; collapse = overconfidence; explosion = instability.
            info["target_ensemble_sigma_mean"] = std_mat.mean().item()
            info["target_ensemble_sigma_std"] = std_mat.std().item()

            # --- Fusion diagnostics (s', a') ---
            # Fused μ is the value signal used for the target
            # Fused σ is the “post-fusion uncertainty”; should not collapse too early
            info["fusion_mu_mean"] = fusion_u.mean().item()
            info["fusion_mu_std"] = fusion_u.std().item()

            info["fusion_sigma_mean"] = fusion_std.mean().item()
            info["fusion_sigma_std"] = fusion_std.std().item()

            # weights: (B, E)  -- contribution of each critic to fused estimate
            eps = 1e-12

            # Dominance: how much the most trusted critic contributes
            # w_max ≈ 1/E  -> equal trust
            # w_max → 1.0  -> single critic dominating (ensemble collapse)
            w_max = fusion_weights.max(dim=1).values  # (B,)

            info["fusion_w_max_mean"] = w_max.mean().item()
            info["fusion_w_max_p95"] = w_max.quantile(0.95).item()

            # Entropy of weights: distribution of trust
            # High entropy  -> distributed trust across critics
            # Low entropy   -> sharp trust concentration
            entropy = -(fusion_weights * (fusion_weights + eps).log()).sum(
                dim=1
            )  # (B,)
            info["fusion_w_entropy_mean"] = entropy.mean().item()
            info["fusion_w_entropy_std"] = entropy.std().item()

            # Effective Ensemble Size
            # N_eff = 1 / sum(w_k^2)
            # ≈ E  -> all critics contributing
            # ≈ 1  -> effectively a single critic
            n_eff = 1.0 / (fusion_weights.pow(2).sum(dim=1) + eps)  # (B,)

            info["fusion_n_eff_mean"] = n_eff.mean().item()
            info["fusion_n_eff_p10"] = n_eff.quantile(0.10).item()

            # --- Target distribution diagnostics ---
            # Drift upward without reward improvement: gamma/reward_scale/instability.
            info["u_target_mean"] = u_target.mean().item()
            info["u_target_std"] = u_target.std().item()

            # Collapse -> overconfident targets; explosion -> noisy targets / unstable critics.
            info["std_target_mean"] = std_target.mean().item()
            info["std_target_std"] = std_target.std().item()

            # --- Critic loss diagnostics (fit quality) ---
            info["critic_loss_total"] = float(np.mean(critic_loss_totals))
            info["critic_loss_totals"] = critic_loss_totals
            # If one critic stays high: “bad apple” critic, poor calibration, or optimizer issue.

            # --- KL diagnostics (more robust than mean loss alone) ---
            # ---- Mean KL across critics (overall fit quality) ----
            kl_mean_per_sample = kl_stack.mean(dim=0)  # (B,1)

            info["kl_mean"] = kl_mean_per_sample.mean().item()
            info["kl_mean_std"] = kl_mean_per_sample.std().item()

            # ---- Max KL across critics (worst critic instability) ----
            # Spikes: distribution mismatch / exploding σ / unstable learning.
            info["kl_max_mean"] = critic_max_per_sample.mean().item()
            info["kl_max_std"] = critic_max_per_sample.std().item()
            info["kl_max_p95"] = critic_max_per_sample.quantile(0.95).item()

            # --- Current ensemble health on replay (s, a) ---
            # σ collapsing while KL remains high => overconfident wrong critics.
            info["current_ensemble_mu_mean"] = float(np.mean(current_mu_means))
            info["current_ensemble_sigma_mean"] = float(np.mean(current_sigma_means))

            # --- PER priority health ---
            # priority_max exploding => PER may over-focus on a few transitions and destabilize training.
            info["priority_mean"] = float(np.mean(priorities))
            info["priority_p95"] = float(np.quantile(priorities, 0.95))
            info["priority_max"] = float(np.max(priorities))

        return info, priorities

    def _update_actor(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        batch_size = len(states)

        actor_q_u_set: list[torch.Tensor] = []
        actor_q_std_set: list[torch.Tensor] = []

        # Track per-critic batch means for “bad apple” detection (like critic update)
        current_mu_means: list[float] = []
        current_sigma_means: list[float] = []

        actions = self.actor_net(states)
        with hlp.evaluating(self.critic_net):
            for critic_net in self.critic_net.critics:
                actor_q_u, actor_q_std = critic_net(states, actions)

                actor_q_u_set.append(actor_q_u)
                actor_q_std_set.append(actor_q_std)

                current_mu_means.append(actor_q_u.mean().item())
                current_sigma_means.append(actor_q_std.mean().item())

        fusion_u_a, fusion_std_a, fusion_weights_a = self._fuse_critic_outputs(
            batch_size, actor_q_u_set, actor_q_std_set
        )

        actor_loss = -fusion_u_a.mean()

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

            # --- Actor-side ensemble diagnostics (s, pi(s)) ---
            u_mat = torch.concat(actor_q_u_set, dim=1)  # (B,E)
            std_mat = torch.concat(actor_q_std_set, dim=1)  # (B,E)

            # Per-sample disagreement across critics on μ under current policy (epistemic spread).
            info["actor_ensemble_mu_mean"] = u_mat.mean().item()
            info["actor_ensemble_mu_std_mean"] = (
                u_mat.std(dim=1, unbiased=False).mean().item()
            )

            # Average predicted uncertainty under current policy; collapse/explosion are red flags.
            info["actor_ensemble_sigma_mean"] = std_mat.mean().item()
            info["actor_ensemble_sigma_std"] = std_mat.std(unbiased=False).item()

            # --- “Bad apple” ensemble drift (across critics, coarse) ---
            # If one critic drifts, these rise even if the per-sample std looks OK.
            info["actor_mu_std_across_critics"] = float(np.std(current_mu_means))
            info["actor_sigma_std_across_critics"] = float(np.std(current_sigma_means))

            # --- Actor-side fusion outputs ---
            info["actor_fusion_mu_mean"] = fusion_u_a.mean().item()
            info["actor_fusion_mu_std"] = fusion_u_a.std(unbiased=False).item()
            info["actor_fusion_sigma_mean"] = fusion_std_a.mean().item()
            info["actor_fusion_sigma_std"] = fusion_std_a.std(unbiased=False).item()

            # --- Fusion weight diagnostics (actor-side) ---
            eps = 1e-12

            # Dominance: near 1 => single critic dominating under actor actions.
            w_max = fusion_weights_a.max(dim=1).values  # (B,)
            info["actor_fusion_w_max_mean"] = w_max.mean().item()
            info["actor_fusion_w_max_p95"] = w_max.quantile(0.95).item()

            # Diversity of trust across critics.
            entropy = -(fusion_weights_a * (fusion_weights_a + eps).log()).sum(
                dim=1
            )  # (B,)
            info["actor_fusion_w_entropy_mean"] = entropy.mean().item()
            info["actor_fusion_w_entropy_std"] = entropy.std(unbiased=False).item()

            # Effective ensemble size: near 1 => effectively single-critic behavio
            n_eff = 1.0 / (fusion_weights_a.pow(2).sum(dim=1) + eps)  # (B,)
            info["actor_fusion_n_eff_mean"] = n_eff.mean().item()
            info["actor_fusion_n_eff_p10"] = n_eff.quantile(0.10).item()

            info["actor_loss"] = actor_loss.item()

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        super().save_models(filepath, filename)
        # Save each ensemble critic optimizer in a single file
        ensemble_optim_state = {
            f"optimizer_{idx}": opt.state_dict()
            for idx, opt in enumerate(self.ensemble_critic_optimizers)
        }
        torch.save(
            ensemble_optim_state,
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth",
        )

    def load_models(self, filepath: str, filename: str) -> None:
        super().load_models(filepath, filename)
        # Load each ensemble critic optimizer from the single file
        ensemble_optim_state = torch.load(
            f"{filepath}/{filename}_ensemble_critic_optimizers.pth"
        )
        for idx, opt in enumerate(self.ensemble_critic_optimizers):
            opt.load_state_dict(ensemble_optim_state[f"optimizer_{idx}"])
