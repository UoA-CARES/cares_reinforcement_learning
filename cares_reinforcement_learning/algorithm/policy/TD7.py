"""
TD7 (TD3 + SALE + Checkpoints + LAP + BC)
-------------------------------------------

Original Paper: https://arxiv.org/pdf/2306.02451
Original Code: https://github.com/sfujim/TD7

TD7 extends TD3 by integrating state-action representation
learning (SALE), policy checkpoints, prioritized replay (LAP),
and an optional behavior cloning term (offline RL).

Core Motivation:
- Value learning from the Bellman equation is a weak and
  non-stationary learning signal.
- Even low-dimensional state spaces can benefit from
  representation learning.
- Actor-critic methods are unstable and prone to
  extrapolation error.

Key Component 1 — SALE (State-Action Learned Embeddings)
---------------------------------------------------------
Learn joint embeddings of state and action by modeling
latent dynamics:

    z_s  = f(s)
    z_sa = g(z_s, a)

Encoders are trained via next-state prediction:

    L = || g(f(s), a) - stopgrad(f(s')) ||^2

Embeddings are:
    • Decoupled from value/policy gradients
    • Normalized (AvgL1Norm)
    • Used as additional inputs to Q and π

Value input:
    Q(z_sa, z_s, s, a)

Policy input:
    π(z_s, s)

This improves feature learning even in low-dimensional control.

Key Component 2 — Clipped Value Targets
----------------------------------------
Expanding state-action inputs can cause extrapolation error.
TD7 clips Bellman targets to the observed value range:

    target = r + γ clip(min(Q1', Q2'), Q_min, Q_max)

This stabilizes value estimates during online learning.

Key Component 3 — Policy Checkpoints
-------------------------------------
Instead of always deploying the latest policy,
TD7 maintains the best-performing checkpoint policy
(measured over evaluation episodes).

Training is batched over assessment episodes:
    collect N steps → train N updates

Evaluation uses the checkpoint policy,
improving robustness to instability.

Key Component 4 — LAP Replay
-----------------------------
Uses Loss-Adjusted Prioritized replay (LAP):

    p(i) ∝ max(|δ_i|^α, 1)

Critic loss uses Huber loss.
Improves sample efficiency and stability.

Key Component 5 — Offline Extension
------------------------------------
For offline RL, adds TD3+BC-style behavior cloning:

    maximize Q(s, π(s)) - λ ||π(s) - a||^2

(λ = 0 in online setting)

Algorithm Summary:
------------------
TD7 = TD3
      + state-action representation learning (SALE)
      + clipped Bellman targets
      + prioritized replay (LAP)
      + policy checkpoints
      + optional behavior cloning (offline)

Key Behaviour:
- Significant sample efficiency gains.
- Strong early performance (300k steps).
- Robust to instability and value overestimation.
- Works in both online and offline regimes.

TD7 is not a minor TD3 tweak — it is a
stability- and representation-focused redesign.
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.algorithm.lossess as loss
import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import TD7Config
from cares_reinforcement_learning.algorithm.schedulers import ExponentialScheduler
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.TD7 import Actor, Critic, Encoder
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation


class TD7(SARLAlgorithm[np.ndarray]):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        encoder_network: Encoder,
        config: TD7Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_actor_net.eval()  # never in training mode - helps with batch/drop out layers
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        self.encoder_net = encoder_network.to(device)
        self.fixed_encoder_net = copy.deepcopy(self.encoder_net).to(self.device)
        self.target_fixed_encoder_net = copy.deepcopy(self.encoder_net).to(self.device)

        self.checkpoint_actor = copy.deepcopy(self.actor_net).to(self.device)
        self.checkpoint_encoder = copy.deepcopy(self.encoder_net).to(self.device)

        self.gamma = config.gamma
        self.tau = config.tau

        self.target_update_freq = config.target_update_rate

        # Checkpoint tracking
        self.max_eps_checkpointing = config.max_eps_checkpointing
        self.steps_before_checkpointing = config.steps_before_checkpointing
        self.reset_weight = config.reset_weight

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0.0
        self.min_target = 0.0

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        # Policy noise
        self.policy_noise_clip = config.policy_noise_clip
        self.policy_noise_scheduler = ExponentialScheduler(
            start_value=config.policy_noise_start,
            end_value=config.policy_noise_end,
            decay_steps=config.policy_noise_decay,
        )
        self.policy_noise = self.policy_noise_scheduler.get_value(0)

        # Action noise
        self.action_noise_scheduler = ExponentialScheduler(
            start_value=config.action_noise_start,
            end_value=config.action_noise_end,
            decay_steps=config.action_noise_decay,
        )
        self.action_noise = self.action_noise_scheduler.get_value(0)

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq

        self.action_num = self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        # Encoder optimiser
        self.encoder_net_optimiser = torch.optim.Adam(
            self.encoder_net.parameters(),
            lr=config.encoder_lr,
            **config.encoder_lr_params,
        )

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        self.actor_net.eval()

        state = observation.vector_state

        with torch.no_grad():
            # Fix: Use modern tensor creation
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)

            if evaluation:
                zs = self.checkpoint_encoder.zs(state_tensor)
                action = self.checkpoint_actor(state_tensor, zs)
            else:
                zs = self.fixed_encoder_net.zs(state_tensor)
                action = self.actor_net(state_tensor, zs)

            action = action.cpu().data.numpy().flatten()

            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(
                    0, scale=self.action_noise, size=self.action_num
                )
                action = action + noise
                action = np.clip(action, -1, 1)

        self.actor_net.train()

        return ActionSample(action=action, source="policy")

    def _calculate_value(self, state: SARLObservation, action: np.ndarray) -> float:  # type: ignore[override]
        # Fix: Use modern tensor creation
        state_tensor = torch.tensor(
            state.vector_state, dtype=torch.float32, device=self.device
        )
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with fnc.evaluating(self.critic_net):
                # Fix: Use proper TD7 critic interface with encodings
                fixed_zs = self.fixed_encoder_net.zs(state_tensor)
                fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, action_tensor)

                q_values_one, q_values_two = self.critic_net(
                    state_tensor, action_tensor, fixed_zsa, fixed_zs
                )
                q_value = torch.minimum(q_values_one, q_values_two)

        return q_value[0].item()

    def _update_encoder(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            next_zs = self.encoder_net.zs(next_states)

        zs = self.encoder_net.zs(states)
        pred_zs = self.encoder_net.zsa(zs, actions)

        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_net_optimiser.zero_grad()
        encoder_loss.backward()
        self.encoder_net_optimiser.step()

        with torch.no_grad():
            # --- SALE / representation health ---
            # If these collapse to ~0 (or explode), your representations are unhealthy even if RL loss looks fine.
            info["encoder_loss"] = encoder_loss.item()

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

            # Tail risk: max per-sample latent prediction error (spots “bad transitions” / distribution shifts)
            per_sample_mse = (pred_zs - next_zs).pow(2).mean(dim=1)
            info["sale_mse_mean"] = per_sample_mse.mean().item()
            info["sale_mse_p95"] = per_sample_mse.quantile(0.95).item()

        return info

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

        q_values_one, q_values_two = self.critic_net(
            states, actions, fixed_zsa, fixed_zs
        )

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        huber_loss_one = loss.calculate_huber_loss(
            td_error_one,
            self.min_priority,
            use_quadratic_smoothing=False,
            use_mean_reduction=False,
        )
        huber_loss_two = loss.calculate_huber_loss(
            td_error_two,
            self.min_priority,
            use_quadratic_smoothing=False,
            use_mean_reduction=False,
        )

        critic_loss_total = (huber_loss_one + huber_loss_two).mean()

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

            # How active is clipping?
            # High clamp_frac means you’re *often* forcing targets into a limited band.
            hit_min = (target_q_values <= self.min_target).float()
            hit_max = (target_q_values >= self.max_target).float()
            info["clip_hit_min_frac"] = hit_min.mean().item()
            info["clip_hit_max_frac"] = hit_max.mean().item()
            info["clip_any_frac"] = torch.maximum(hit_min, hit_max).mean().item()

            # How much is clipping changing the target values?
            clip_delta = (target_q_values - target_q_values_clipped).abs()
            info["clip_delta_abs_mean"] = clip_delta.mean().item()
            info["clip_delta_abs_p95"] = clip_delta.quantile(0.95).item()

            # Track the clip band you are using right now
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
            info["critic_loss_total"] = critic_loss_total.item()

        return info, priorities

    # Weights is set for methods like MAPERTD3 that use weights in the actor update
    def _update_actor(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with fnc.evaluating(self.encoder_net):
            fixed_zs = self.fixed_encoder_net.zs(states)

        actions = self.actor_net(states, fixed_zs)
        fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, actions)

        with fnc.evaluating(self.critic_net):
            actor_q_values_one, actor_q_values_two = self.critic_net(
                states, actions, fixed_zsa, fixed_zs
            )

        # Concatenate both Q-values then take mean (like reference TD7)
        actor_q_values = torch.cat([actor_q_values_one, actor_q_values_two], dim=1)
        actor_loss = -actor_q_values.mean()

        # ---------------------------------------------------------
        # Deterministic Policy Gradient Strength (∇a Q(s,a))  [TD7 actor]
        # ---------------------------------------------------------
        # Same interpretation as TD3, but note Q depends on SALE features too.
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
            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_values.mean().item()
            info["actor_q_std"] = actor_q_values.std(unbiased=False).item()
            info["qf_pi_gap_abs_mean"] = (
                (actor_q_values_one - actor_q_values_two).abs().mean().item()
            )

        return info

    def update_from_batch(
        self,
        episode_context: EpisodeContext,
        memory: SARLMemoryBuffer,
        indices: np.ndarray,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        self.policy_noise = self.policy_noise_scheduler.get_value(
            episode_context.training_step
        )

        self.action_noise = self.action_noise_scheduler.get_value(
            episode_context.training_step
        )

        info["policy_noise"] = float(self.policy_noise)
        info["action_noise"] = float(self.action_noise)

        encoder_info = self._update_encoder(
            states_tensor, actions_tensor, next_states_tensor
        )
        info |= encoder_info

        # Update the Critic
        critic_info, priorities = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        # Update the Priorities
        if self.use_per_buffer:
            memory.update_priorities(indices, priorities)

        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor(states_tensor, weights_tensor)
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            # Update target network params
            self.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
            self.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

            self.soft_update_params(
                self.fixed_encoder_net, self.target_fixed_encoder_net, self.tau
            )
            self.soft_update_params(self.encoder_net, self.fixed_encoder_net, self.tau)

            memory.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        return info

    def _train_policy(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        self.learn_counter += 1

        # Use the helper to sample and prepare tensors in one step
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            _,  # extras ignored
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        info = self.update_from_batch(
            episode_context,
            memory_buffer,
            indices,
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
            weights_tensor,
        )

        return info

    def _train_and_reset(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        # Log pre-reset state (useful for debugging / plotting)
        info["pre_reset_best_min_return"] = float(self.best_min_return)
        info["pre_reset_max_eps_before_update"] = int(self.max_eps_before_update)
        info["pre_reset_timesteps_since_update"] = int(self.timesteps_since_update)
        info["pre_reset_eps_since_update"] = int(self.eps_since_update)

        # Track whether we hit the checkpointing regime switch during this training burst
        checkpoint_regime_switched = False

        for _ in range(self.timesteps_since_update):
            if self.learn_counter == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_checkpointing
                checkpoint_regime_switched = True

            info = self._train_policy(memory_buffer, episode_context)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

        # Log post-reset state + event flag
        info["checkpoint_regime_switched"] = float(checkpoint_regime_switched)
        info["post_reset_best_min_return"] = float(self.best_min_return)
        info["post_reset_max_eps_before_update"] = int(self.max_eps_before_update)

        return info

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        episode_steps = episode_context.episode_steps
        episode_return = episode_context.episode_reward
        episode_done = episode_context.episode_done

        if not episode_done:
            return info

        # -------------------------
        # TD7 gate bookkeeping
        # -------------------------
        self.eps_since_update += 1
        self.timesteps_since_update += episode_steps
        self.min_return = min(self.min_return, episode_return)

        # -------------------------
        # Episode-level logging (always)
        # -------------------------
        info["episode_return"] = float(episode_return)
        info["episode_steps"] = int(episode_steps)

        # "Worst-case since last update" tracking (TD7's gate signal)
        info["min_return_window"] = float(self.min_return)
        info["best_min_return"] = float(self.best_min_return)

        # Cadence / counters
        info["eps_since_update"] = int(self.eps_since_update)
        info["timesteps_since_update"] = int(self.timesteps_since_update)
        info["max_eps_before_update"] = int(self.max_eps_before_update)
        info["learn_counter"] = int(self.learn_counter)

        # Decision flags (exactly one of these becomes 1.0 on episode end)
        reset_triggered = self.min_return < self.best_min_return
        accept_triggered = (
            self.eps_since_update == self.max_eps_before_update
        ) and not reset_triggered

        info["reset_triggered"] = float(reset_triggered)
        info["accept_triggered"] = float(accept_triggered)

        # How far from the gate are we?
        # Negative reset_margin => will reset; positive means "safe" above best_min_return.
        info["reset_margin"] = float(self.min_return - self.best_min_return)

        # -------------------------
        # TD7 decisions
        # -------------------------
        if reset_triggered:
            train_info = self._train_and_reset(memory_buffer, episode_context)
            info.update(train_info)

        elif accept_triggered:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor_net.state_dict())
            self.checkpoint_encoder.load_state_dict(self.encoder_net.state_dict())

            train_info = self._train_and_reset(memory_buffer, episode_context)
            info.update(train_info)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "encoder": self.encoder_net.state_dict(),
            "target_actor": self.target_actor_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "target_fixed_encoder": self.target_fixed_encoder_net.state_dict(),
            "fixed_encoder": self.fixed_encoder_net.state_dict(),
            "checkpoint_actor": self.checkpoint_actor.state_dict(),
            "checkpoint_encoder": self.checkpoint_encoder.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "encoder_optimizer": self.encoder_net_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
            "policy_noise": self.policy_noise,
            "action_noise": self.action_noise,
            # Add value tracking
            "max": self.max,
            "min": self.min,
            "max_target": self.max_target,
            "min_target": self.min_target,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.target_actor_net.load_state_dict(checkpoint["target_actor"])

        self.critic_net.load_state_dict(checkpoint["critic"])
        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        # Load encoder networks
        self.encoder_net.load_state_dict(checkpoint["encoder"])
        self.target_fixed_encoder_net.load_state_dict(
            checkpoint["target_fixed_encoder"]
        )
        self.fixed_encoder_net.load_state_dict(checkpoint["fixed_encoder"])
        self.checkpoint_actor.load_state_dict(checkpoint["checkpoint_actor"])
        self.checkpoint_encoder.load_state_dict(checkpoint["checkpoint_encoder"])

        # Load optimizers
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        self.encoder_net_optimiser.load_state_dict(checkpoint["encoder_optimizer"])

        # Load training state
        self.learn_counter = checkpoint.get("learn_counter", 0)
        self.policy_noise = checkpoint.get("policy_noise", self.policy_noise)
        self.action_noise = checkpoint.get("action_noise", self.action_noise)

        # Load value tracking
        self.max = checkpoint.get("max", -1e8)
        self.min = checkpoint.get("min", 1e8)
        self.max_target = checkpoint.get("max_target", 0)
        self.min_target = checkpoint.get("min_target", 0)

        logging.info("models, optimisers, and training state have been loaded...")
