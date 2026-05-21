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
	  𝜆 gradually decreases (vice versa).
	- Gradient-ascent (GA) updates guarantee 
	  convergence to 𝜆*.
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

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.configuration import (
    SACLagConfig,
    LagrangeMultiplierConfig,
)
from cares_reinforcement_learning.algorithm.policy.SAC import SAC
from cares_reinforcement_learning.networks.common import (
    EnsembleCritic,
    QNetwork,
    TanhGaussianPolicy,
    TwinQNetwork,
)


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
            raise ValueError(f"Unknown update method: {self.update_method}")

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

        super.__init__(actor_network, critic_network, config, device)

        self.cost_gamma = config.cost_gamma

        self.cost_critic_net = cost_critic_network.to(self.device)

        self.target_cost_critic_net = copy.deepcopy(
            self.cost_critic_net
        ).to(self.device)
        self.target_cost_critic_net.eval()

        self.cost_critic_optimiser = torch.optim.Adam(
            self.cost_critic_net.parameters(),
            lr=config.cost_critic_lr,
            **config.cost_critic_lr_params
        )

        self.lagrange_multiplier = LagrangeMultiplier(
            config.lagrange_multiplier, device
        )

    @property
    def lagrange_multiplier():
        pass

    def _update_cost_critic():
        pass

    def _update_actor_alpha():
        pass

    def update_from_batch():
        pass

    def update_target_networks():
        pass

    def train():
        pass

    def save_models():
        pass

    def load_models():
        pass