"""
TD7SIGReg (TD7 + SIGReg representation regularisation)
-------------------------------------------------------

TD7 Paper: https://arxiv.org/pdf/2306.02451
SIGReg (LeJEPA) Paper: https://arxiv.org/abs/2511.08544

TD7SIGReg is TD7 with SIGReg used to keep representations well-conditioned.

Encoder (always on)
-------------------
TD7:
    z_s = AvgL1Norm(f(s))
    L_enc = || g(z_s, a) - stopgrad(z_s') ||^2

TD7SIGReg:
    z_s = f(s)                          (no AvgL1Norm)
    L_pred = || g(z_s, a) - stopgrad(z_s') ||^2
    L_enc  = (1 - λ_enc) * L_pred + λ_enc * SIGReg(z_s)

λ_enc = encoder_sigreg_lambda (default 0.05, as in LeJEPA).

SIGReg (Sketched Isotropic Gaussian Regularisation) pushes the
batch distribution of an embedding towards an isotropic Gaussian by
matching the empirical characteristic function of random 1D projections
to that of N(0, 1).

Actor / critic ablations (all off by default -> TD7 actor/critic)
------------------------------------------------------------------
* actor_use_avg_l1_norm / critic_use_avg_l1_norm:
    toggle AvgL1Norm on the feature-layer output.

* SIGReg on two representation sites per network:
    feature     - feature-layer output (the tensor fed to AvgL1Norm, pre-norm)
    penultimate - output of the last Linear of *_body_config (pre-activation)
  with independent lambdas {actor,critic}_{feature,penultimate}_sigreg_lambda.
  For the twin critic the SIGReg term is averaged over Q1 and Q2.

* {actor,critic}_sigreg_mode - how SIGReg is combined with the RL loss:
    additive: L = L_rl + Σ λ_i S_i
    convex:   L = (1 - Σ λ_i) L_rl + Σ λ_i S_i
"""

from typing import Any, cast

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.algorithm.lossess as loss
from cares_reinforcement_learning.algorithm.configurations import TD7SIGRegConfig
from cares_reinforcement_learning.algorithm.policy.TD7 import TD7
from cares_reinforcement_learning.algorithm.sigreg import SIGReg
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.TD7SIGReg import Actor, Critic, Encoder


class TD7SIGReg(TD7):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        encoder_network: Encoder,
        config: TD7SIGRegConfig,
        device: torch.device,
    ):
        # TD7SIGReg's Actor/Critic share TD7's call interface (forward) and add
        # forward_with_features for the SIGReg sites.
        super().__init__(
            actor_network=actor_network,  # type: ignore[arg-type]
            critic_network=critic_network,  # type: ignore[arg-type]
            encoder_network=encoder_network,
            config=config,
            device=device,
        )

        self.encoder_sigreg_lambda = config.encoder_sigreg_lambda

        self.actor_feature_sigreg_lambda = config.actor_feature_sigreg_lambda
        self.actor_penultimate_sigreg_lambda = config.actor_penultimate_sigreg_lambda
        self.actor_sigreg_mode = config.actor_sigreg_mode

        self.critic_feature_sigreg_lambda = config.critic_feature_sigreg_lambda
        self.critic_penultimate_sigreg_lambda = config.critic_penultimate_sigreg_lambda
        self.critic_sigreg_mode = config.critic_sigreg_mode

        self.sigreg = SIGReg(
            knots=config.sigreg_knots,
            t_max=config.sigreg_t_max,
            num_slices=config.sigreg_num_slices,
        ).to(device)

    # Typed views of the online networks (exposes forward_with_features)
    @property
    def sigreg_actor_net(self) -> Actor:
        return cast(Actor, self.actor_net)

    @property
    def sigreg_critic_net(self) -> Critic:
        return cast(Critic, self.critic_net)

    @staticmethod
    def _combine_sigreg(
        base_loss: torch.Tensor,
        terms: list[tuple[float, torch.Tensor]],
        mode: str,
    ) -> torch.Tensor:
        """Combine an RL loss with weighted SIGReg terms (additive or convex)."""
        if not terms:
            return base_loss

        reg = sum(lam * value for lam, value in terms)
        if mode == "convex":
            lambda_sum = sum(lam for lam, _ in terms)
            return (1.0 - lambda_sum) * base_loss + reg
        if mode == "additive":
            return base_loss + reg
        raise ValueError(f"Unknown SIGReg combination mode: {mode}")

    def _sigreg_terms(
        self,
        features: dict[str, Any],
        feature_lambda: float,
        penultimate_lambda: float,
        prefix: str,
        info: dict[str, Any],
    ) -> list[tuple[float, torch.Tensor]]:
        """Compute enabled SIGReg terms; each site may be a tensor or a list (twin critic)."""
        terms: list[tuple[float, torch.Tensor]] = []

        for site, lam in (
            ("feature", feature_lambda),
            ("penultimate", penultimate_lambda),
        ):
            values = features[site]
            values = values if isinstance(values, list) else [values]

            if lam > 0.0:
                sigreg_value = torch.stack([self.sigreg(v) for v in values]).mean()
                terms.append((lam, sigreg_value))
                info[f"{prefix}_{site}_sigreg_loss"] = sigreg_value.item()

            with torch.no_grad():
                stacked = torch.cat(values, dim=0)
                info[f"{prefix}_{site}_abs_mean"] = stacked.abs().mean().item()
                info[f"{prefix}_{site}_std"] = stacked.std(unbiased=False).item()

        return terms

    def _update_encoder(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            next_zs = self.encoder_net.zs(next_states)

        zs = self.encoder_net.zs(states)
        pred_zs = self.encoder_net.zsa(zs, actions)

        pred_loss = F.mse_loss(pred_zs, next_zs)
        sigreg_loss = self.sigreg(zs)

        encoder_loss = (
            1.0 - self.encoder_sigreg_lambda
        ) * pred_loss + self.encoder_sigreg_lambda * sigreg_loss

        self.encoder_net_optimiser.zero_grad()
        encoder_loss.backward()
        self.encoder_net_optimiser.step()

        with torch.no_grad():
            # --- SALE / representation health ---
            info["encoder_loss"] = encoder_loss.item()
            info["sale_pred_loss"] = pred_loss.item()
            info["encoder_sigreg_loss"] = sigreg_loss.item()
            info["encoder_sigreg_lambda"] = float(self.encoder_sigreg_lambda)

            info["zs_abs_mean"] = zs.abs().mean().item()
            info["zs_std"] = zs.std(unbiased=False).item()
            info["pred_zs_abs_mean"] = pred_zs.abs().mean().item()
            info["pred_zs_std"] = pred_zs.std(unbiased=False).item()

            # Cosine similarity: are we predicting the right *direction* in latent space?
            eps = 1e-12
            cos = (pred_zs * next_zs).sum(dim=1) / (
                pred_zs.norm(dim=1) * next_zs.norm(dim=1) + eps
            )
            info["sale_cos_mean"] = cos.mean().item()
            info["sale_cos_p05"] = cos.quantile(0.05).item()

            # Tail risk: max per-sample latent prediction error
            per_sample_mse = (pred_zs - next_zs).pow(2).mean(dim=1)
            info["sale_mse_mean"] = per_sample_mse.mean().item()
            info["sale_mse_p95"] = per_sample_mse.quantile(0.95).item()

        return info

    # Copy of TD7._update_critic with optional SIGReg on the critic representations
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
            fixed_target_zs = self.target_fixed_encoder_net.zs(next_states)

            next_actions = self.target_actor_net(next_states, fixed_target_zs)

            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )

            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            fixed_target_zsa = self.target_fixed_encoder_net.zsa(
                fixed_target_zs, next_actions
            )

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions, fixed_target_zsa, fixed_target_zs
            )

            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            # TD7 value clipping
            target_q_values_clipped = target_q_values.clamp(
                self.min_target, self.max_target
            )

            q_target = rewards + self.gamma * (1 - dones) * target_q_values_clipped

            # tracked range (used to set next min_target/max_target later)
            self.max = max(self.max, float(q_target.max()))
            self.min = min(self.min, float(q_target.min()))

            fixed_zs = self.fixed_encoder_net.zs(states)
            fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, actions)

        q_values_one, q_values_two, critic_features = (
            self.sigreg_critic_net.forward_with_features(
                states, actions, fixed_zsa, fixed_zs
            )
        )

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        huber_loss_one = loss.calculate_huber_loss(
            td_error_one,
            self.min_priority,
            use_huber_offset=False,
            use_mean_reduction=False,
        )
        huber_loss_two = loss.calculate_huber_loss(
            td_error_two,
            self.min_priority,
            use_huber_offset=False,
            use_mean_reduction=False,
        )

        critic_td_loss = (huber_loss_one + huber_loss_two).mean()

        sigreg_terms = self._sigreg_terms(
            critic_features,
            self.critic_feature_sigreg_lambda,
            self.critic_penultimate_sigreg_lambda,
            "critic",
            info,
        )
        critic_loss_total = self._combine_sigreg(
            critic_td_loss, sigreg_terms, self.critic_sigreg_mode
        )

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
            # --- Target policy smoothing diagnostics (TD3/TD7) ---
            info["target_noise_abs_mean"] = target_noise.abs().mean().item()
            info["target_noise_clip_frac"] = (
                (target_noise.abs() >= self.policy_noise_clip).float().mean().item()
            )

            # --- SALE feature inputs (sanity) ---
            info["fixed_zs_abs_mean"] = fixed_zs.abs().mean().item()
            info["fixed_zsa_abs_mean"] = fixed_zsa.abs().mean().item()

            # --- Target critics + clipping diagnostics (TD7-specific) ---
            info["target_q1_mean"] = target_q_values_one.mean().item()
            info["target_q2_mean"] = target_q_values_two.mean().item()
            info["target_q_min_mean"] = target_q_values.mean().item()

            hit_min = (target_q_values <= self.min_target).float()
            hit_max = (target_q_values >= self.max_target).float()
            info["clip_hit_min_frac"] = hit_min.mean().item()
            info["clip_hit_max_frac"] = hit_max.mean().item()
            info["clip_any_frac"] = torch.maximum(hit_min, hit_max).mean().item()

            clip_delta = (target_q_values - target_q_values_clipped).abs()
            info["clip_delta_abs_mean"] = clip_delta.mean().item()
            info["clip_delta_abs_p95"] = clip_delta.quantile(0.95).item()

            info["clip_min_target"] = float(self.min_target)
            info["clip_max_target"] = float(self.max_target)
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # --- Current critics health ---
            info["q1_mean"] = q_values_one.mean().item()
            info["q2_mean"] = q_values_two.mean().item()
            info["q_twin_gap_abs_mean"] = (
                (q_values_one - q_values_two).abs().mean().item()
            )

            # --- TD error diagnostics (fit quality + tail risk) ---
            td_abs_max = torch.maximum(td_error_one, td_error_two).squeeze(1)  # (B,)
            info["td_abs_mean"] = td_abs_max.mean().item()
            info["td_abs_p95"] = td_abs_max.quantile(0.95).item()
            info["td_abs_max"] = td_abs_max.max().item()

            # --- Losses (LAP/Huber) ---
            info["critic_loss_one"] = huber_loss_one.mean().item()
            info["critic_loss_two"] = huber_loss_two.mean().item()
            info["critic_td_loss"] = critic_td_loss.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities

    # Copy of TD7._update_actor with optional SIGReg on the actor representations
    def _update_actor(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with fnc.evaluating(self.encoder_net):
            fixed_zs = self.fixed_encoder_net.zs(states)

        actions, actor_features = self.sigreg_actor_net.forward_with_features(
            states, fixed_zs
        )
        fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, actions)

        with fnc.evaluating(self.critic_net):
            actor_q_values_one, actor_q_values_two = self.critic_net(
                states, actions, fixed_zsa, fixed_zs
            )

        # Concatenate both Q-values then take mean (like reference TD7)
        actor_q_values = torch.cat([actor_q_values_one, actor_q_values_two], dim=1)
        actor_q_loss = -actor_q_values.mean()

        # Deterministic Policy Gradient Strength (∇a Q(s,a)) - Q term only
        dq_da = torch.autograd.grad(
            outputs=actor_q_loss,
            inputs=actions,
            retain_graph=True,  # because we do backward(actor_loss) next
            create_graph=False,  # diagnostic only
            allow_unused=False,
        )[0]
        with torch.no_grad():
            info["dq_da_abs_mean"] = dq_da.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da.norm(dim=1).quantile(0.95).item()

        sigreg_terms = self._sigreg_terms(
            actor_features,
            self.actor_feature_sigreg_lambda,
            self.actor_penultimate_sigreg_lambda,
            "actor",
            info,
        )
        actor_loss = self._combine_sigreg(
            actor_q_loss, sigreg_terms, self.actor_sigreg_mode
        )

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        with torch.no_grad():
            # --- Action health ---
            info["pi_action_mean"] = actions.mean().item()
            info["pi_action_std"] = actions.std(unbiased=False).item()
            info["pi_action_abs_mean"] = actions.abs().mean().item()
            info["pi_action_saturation_frac"] = (
                (actions.abs() > 0.95).float().mean().item()
            )

            # --- SALE inputs to policy (representation scale sanity) ---
            info["actor_zs_abs_mean"] = fixed_zs.abs().mean().item()
            info["actor_zs_std"] = fixed_zs.std(unbiased=False).item()

            # --- On-policy critic signal ---
            info["actor_q_loss"] = actor_q_loss.item()
            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_values.mean().item()
            info["actor_q_std"] = actor_q_values.std(unbiased=False).item()
            info["qf_pi_gap_abs_mean"] = (
                (actor_q_values_one - actor_q_values_two).abs().mean().item()
            )

        return info
