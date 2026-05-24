"""
SAC-Lag (Soft Actor-Critic with Lagrange multipliers)
-----------------------------------------------------

Original Paper: https://arxiv.org/abs/2510.17564v1

SAC-Lag extends Soft Actor-Critic (SAC) to handle safety
constraints by augmenting the actor objective with a
penalty term weighted by a Lagrange multiplier 𝜆.
This is a Lagrangian relaxation of the underlying CMDP.

Core Idea:
- Enforces constraints asymptotically.
- Penalises constraint violations at each step.
- 𝜆 governs the trade-off between return and cost.

Objective (primal problem):
    max_θ  J^R(π_θ)
    s.t.   J^Ci(π_θ) ≤ d_i,   i = 1, ..., m
where:
    J^R(π_θ)  = expected discounted return
    J^Ci(π_θ) = expected discounted cost for constraint i
    d_i       = cost limit for constraint i

Lagrangian Relaxation (dual form):
    min_𝜆 max_θ [ J^R(π_θ) - Σ 𝜆_i (J^Ci(π_θ) - d_i) ]
where:
    𝜆 ≥ 0

Lagrange Multiplier Updates:
    Option 1 — Fixed:
    - 𝜆 is manually set to a constant before training.
    Option 2 — Gradient Ascent (GA):
    - 𝜆_{k+1}= ( 𝜆__k + η · ξ )+
      where η is a step-size and (·)+ projects to ℝ+.
    Option 3 — PID-controlled:
    - 𝜆_{k+1}= ( K_P·ξ_k + K_I·I_k + K_D·∂_k )+
      where I_k = (I_{k-1} + ξ_k)+ is the accumulated
      violation and ∂_k is the derivative of the cost.

Key Behaviours:
    Fixed 𝜆:
        - If the optimal  𝜆* is known, the solution is
          equivalent to that of the primal CMDP.
        - Reward trajectory is gradual and conservative.
    Automated lambda:
        - If the agent violates fewer constraints,
        - 𝜆 gradually decreases (vice versa).
        - Gradient-ascent (GA) updates guarantee 
        - convergence to 𝜆*.
        - PID updates stabilize 𝜆 trajectory.

Limitations:
- Finding the optimal 𝜆* is computationally expensive and highly
  task‑dependent.
- GA updates often produce oscillatory lambda behavior.
- PID updates introduce three extra hyperparameters and 
  do not guarantee convergence to 𝜆*.

Advantages:
- Minimal overhead compared to vanilla SAC.
- Achieves performance close to the unconstrained optimum.

SAC_lag = SAC + Cost critic +  Lagrange multiplier
"""


import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.configuration import (
    SACLagConfig,
    LagrangeMultiplierConfig,
)
from cares_reinforcement_learning.algorithm.policy.SAC import SAC
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.common import (
    EnsembleCritic,
    QNetwork,
    TanhGaussianPolicy,
    TwinQNetwork,
)
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservationTensors


# TODO: Move to right place
class LagrangeMultiplier:
    def __init__(
        self,
        config: LagrangeMultiplierConfig,
        device: torch.device
    ) -> None:

        self.value = torch.tensor(
            config.init, dtype=torch.float32
        ).to(device)
        self.upper_bound = config.upper_bound
        self.update_method = config.update_method

        self.cost_limit = config.cost_limit  # not inherent but associated

        if self.update_method == "gradient_ascent":
            self.value.requires_grad_(True)
            self.optimiser = torch.optim.Adam(
                [self.value], lr=config.lr, **config.lr_params
            )

        # PID controller
        elif self.update_method == "pid_controller":
            self._pid_kp = config.pid_kp
            self._pid_ki = config.pid_ki
            self._pid_kd = config.pid_kd
            self._integral_max = config.integral_max
            self._integral: float = 0.0
            self._derivative: float = 0.0
            self._prev_error: float | None = None

    def update(
        self,
        mean_episode_cost: float,
        dt: float = 1.0
    ) -> dict[str, Any]:

        if self.update_method == "fixed":
            return {"lagrange_multiplier_value": self.value.item()}

        elif self.update_method == "gradient_ascent":
            return self._update_via_gradient_ascent(mean_episode_cost)

        elif self.update_method == "pid_controller":
            return self._update_via_pid_controller(mean_episode_cost, dt)
        else:
            raise NotImplementedError(
                f"Unknown update method: {self.update_method}"
                'Available update methods are: "fixed", "gradient_ascent", "pid_controller".'
            )

    def _update_via_gradient_ascent(
        self,
        mean_episode_cost: float
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        loss = - self.value * (mean_episode_cost - self.cost_limit)

        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        with torch.no_grad():
            self.value.clamp_(min=0.0, max=self.upper_bound)
            info["lagrange_multiplier_value"] = self.value.item()
            info["lagrange_multiplier_loss"] = loss.item()

        return info

    def _update_via_pid_controller(
        self,
        mean_episode_cost: float,
        dt: float
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        error: float = mean_episode_cost - self.cost_limit

        self._integral += error * dt
        self._integral = max(
            -self._integral_max, min(self._integral, self._integral_max)  # Windup clamp
        )

        if self._prev_error is None:
            self._derivative = 0
        else:
            self._derivative = (error - self._prev_error) / dt
        self._prev_error = error

        delta: float = (
            self._pid_kp * error
            + self._pid_ki * self._integral
            + self._pid_kd * self._derivative
        )

        with torch.no_grad():
            self.value.add_(delta)
            self.value.clamp_(min=0.0, max=self.upper_bound)

            info["lagrange_multiplier_value"] = self.value.item()
            info["lagrange_multiplier_pid_error"] = error
            info["lagrange_multiplier_pid_integral"] = self._integral
            info["lagrange_multiplier_pid_derivative"] = self._derivative
            info["lagrange_multiplier_pid_delta"] = delta

        return info


class SACLag(SAC):
    def __init__(
        self,
        actor_network: TanhGaussianPolicy,
        critic_network: TwinQNetwork | EnsembleCritic,
        cost_critic_network: QNetwork,
        config: SACLagConfig,
        device: torch.device,
    ) -> None:

        super().__init__(actor_network, critic_network, config, device)

        self.cost_scale = config.cost_scale
        self.cost_gamma = config.cost_gamma

        self.cost_critic_net = cost_critic_network.to(self.device)
        self.target_cost_critic_net = copy.deepcopy(
            self.cost_critic_net
        ).to(self.device)
        self.target_cost_critic_net.eval()

        self.cost_critic_net_optimiser = torch.optim.Adam(
            self.cost_critic_net.parameters(),
            lr=config.cost_critic_lr,
            **config.cost_critic_lr_params
        )

        self._lagrange_multiplier = LagrangeMultiplier(
            config.lagrange_multiplier, device
        )

    @property
    def lagrange_multiplier(self) -> torch.Tensor:
        return self._lagrange_multiplier.value.detach()

    def _update_cost_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        costs: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:

        info: dict[str, Any] = {}

        with torch.no_grad():
            with fnc.evaluating(self.actor_net):
                next_actions, _, _ = self.actor_net(next_states)

            next_q_values = self.target_cost_critic_net(next_states, next_actions)

            q_target = (
                costs * self.cost_scale
                + self.cost_gamma * (1 - dones) * next_q_values
            )

        q_values = self.cost_critic_net(states, actions)

        td_error = (q_values - q_target).abs()

        critic_loss = F.mse_loss(q_values, q_target, reduction="none")
        critic_loss = (critic_loss * weights).mean()

        self.cost_critic_net_optimiser.zero_grad()
        critic_loss.backward()
        self.cost_critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            td_error
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        with torch.no_grad():
            # --- Critic values ---
            info["cost_q_mean"] = q_values.mean().item()
            info["cost_next_q_mean"] = next_q_values.mean().item()

            # --- Target critic diagnostics ---
            info["cost_q_target_mean"] = q_target.mean().item()
            info["cost_q_target_std"] = q_target.std().item()

            # --- TD error diagnostics (Bellman fit quality) ---
            td = q_values - q_target
            info["cost_td_mean"] = td.mean().item()
            info["cost_td_std"] = td.std().item()
            info["cost_td_abs_mean"] = td.abs().mean().item()

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["cost_critic_loss"] = critic_loss.item()

        return info, priorities

    def _update_actor_alpha(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        pi, log_pi, _ = self.actor_net(states)

        with fnc.evaluating(self.critic_net), fnc.evaluating(self.cost_critic_net):
            qf_pi_one, qf_pi_two = self.critic_net(states, pi)
            cost_qf_pi = self.cost_critic_net(states, pi)

        min_qf_pi = torch.minimum(qf_pi_one, qf_pi_two)

        actor_loss = - (
            min_qf_pi
            - self.lagrange_multiplier * (cost_qf_pi - self._lagrange_multiplier.cost_limit)
            - (self.alpha * log_pi)
        ).mean() / (1 + self.lagrange_multiplier)  # Regularise gradients

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

        # update the temperature (alpha)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        with torch.no_grad():
            # --- Policy entropy diagnostics (exploration health) ---
            # log_pi more negative -> higher entropy (more stochastic). Less negative -> lower entropy (more deterministic).
            info["log_pi_mean"] = log_pi.mean().item()
            info["log_pi_std"] = log_pi.std().item()

            # --- Action magnitude/saturation (tanh policies) ---
            # High saturation fraction can indicate the policy is slamming bounds; may reduce effective gradients.
            info["pi_action_abs_mean"] = pi.abs().mean().item()
            info["pi_action_std"] = pi.std().item()
            info["pi_action_saturation_frac"] = (pi.abs() > 0.95).float().mean().item()

            # --- On-policy critic signal ---
            # min_qf_pi_mean should generally increase as the policy improves (higher value actions under the policy).
            info["min_qf_pi_mean"] = min_qf_pi.mean().item()

            # --- Twin critics disagreement at policy actions (more relevant than replay actions) ---
            # Large gap here means critics disagree on what the current policy is doing (can destabilize actor updates).
            info["qf_pi_gap_abs_mean"] = (qf_pi_one - qf_pi_two).abs().mean().item()

            # --- Entropy gap (alpha tuning health) ---
            # entropy_gap ~ 0 means entropy matches target.
            # > 0: entropy too low -> alpha should increase; < 0: entropy too high -> alpha should decrease.
            entropy_gap = -(log_pi + self.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            # --- Losses and temperature ---
            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()
            info["log_alpha"] = self.log_alpha.item()

            # --- Lagrange multiplier ---
            info["lagrange_multiplier"] = self.lagrange_multiplier.item()

        return info

    def update_from_episode(
        self,
        mean_ep_cost: float,
    ) -> dict[str, Any]:
        return self._lagrange_multiplier.update(mean_ep_cost)

    def update_from_batch(
        self,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        costs_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:

        self.learn_counter += 1

        info: dict[str, Any] = {}

        # Update the reward critic
        critic_info, priorities = self._update_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        # Update the cost critic
        cost_critic_info, cost_priorities = self._update_cost_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            costs_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= cost_critic_info

        combined_priorities = np.maximum(priorities, cost_priorities)  # could try weighted average

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(
                observation_tensor.vector_state_tensor, weights_tensor
            )
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            self.update_target_networks()

        return info, combined_priorities

    def update_target_networks(self) -> None:
        self.soft_update_params(
            self.critic_net,
            self.target_critic_net,
            self.tau
        )
        self.soft_update_params(
            self.cost_critic_net,
            self.target_cost_critic_net,
            self.tau
        )

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            costs_tensor,
            next_observation_tensor,
            dones_tensor,
            weights_tensor,
            _,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        info, priorities = self.update_from_batch(
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            costs_tensor=costs_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
            weights_tensor=weights_tensor,
        )

        # Update the Priorities
        if self.use_per_buffer:
            memory_buffer.update_priorities(indices, priorities)

        # Update Lagrange Multiplier
        if episode_context.episode_done:
            mean_episode_cost = (
                episode_context.episode_cost / episode_context.episode_steps
            )
            info |= self.update_from_episode(
                mean_episode_cost
            )

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "cost_critic": self.cost_critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "target_cost_critic": self.target_cost_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "cost_critic_optimizer": self.cost_critic_net_optimiser.state_dict(),
            # Save log_alpha as a float, not a numpy array
            "log_alpha": float(self.log_alpha.detach().cpu().item()),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "lagrange_multiplier": self._lagrange_multiplier,  # the entire object
            "learn_counter": int(self.learn_counter),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])
        self.cost_critic_net.load_state_dict(checkpoint["cost_critic"])

        self.target_critic_net.load_state_dict(checkpoint["target_critic"])
        self.target_cost_critic_net.load_state_dict(checkpoint["target_cost_critic"])
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        self.cost_critic_net_optimiser.load_state_dict(checkpoint["cost_critic_optimizer"])

        # Restore log_alpha from float
        self.log_alpha.data = torch.tensor(checkpoint["log_alpha"]).to(self.device)
        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])
        self.learn_counter = checkpoint.get("learn_counter", 0)

        # Restore the entire instance of LagrangeMultiplier
        self._lagrange_multiplier = checkpoint["lagrange_multiplier"]

        logging.info("models, optimisers, and training state have been loaded...")