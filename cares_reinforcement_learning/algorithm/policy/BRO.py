"""
BRO (Bigger, Regularized, Optimistic)
------------------------------------

Original Paper: https://arxiv.org/abs/2405.16158
Original Code: https://github.com/naumix/BiggerRegularizedOptimistic

BRO extends Soft Actor-Critic (SAC) with larger residual networks,
distributional value learning, regularization, and optimistic exploration
to improve sample efficiency and performance in continuous control.

Core Idea:
- Scale actor and critic networks using BroNet residual architectures.
- Maintain network plasticity through weight decay and periodic resets.
- Learn return distributions using twin quantile critics.
- Explore using an optimistic policy guided by critic disagreement.

Key Component 1 — Bigger, Regularized Networks
-----------------------------------------------
BRO uses residual multilayer perceptrons with LayerNorm (BroNet)
to support larger actor and critic architectures.

Regularization includes:
    - AdamW optimization with weight decay.
    - Periodic network and optimizer resets.
    - Retaining the replay buffer across resets.

These mechanisms help maintain learning capacity during training.

Key Component 2 — Distributional Value Learning
------------------------------------------------
Each critic predicts N quantiles of the return distribution:

    Z1(s,a) = {q1_1, ..., q1_N}
    Z2(s,a) = {q2_1, ..., q2_N}

BRO combines the critics using:

    Z(s,a) = (Z1 + Z2)/2 - p * |Z1 - Z2|/2

where:
    p controls pessimism (default 0).

Unlike SAC's clipped double-Q learning, BRO defaults to averaging
the critics rather than taking their minimum.

Critics are trained using quantile Huber regression.

Key Component 3 — Optimistic Exploration
-----------------------------------------
BRO maintains two separate policies:

    Standard actor:
        Learns to maximize expected return with SAC entropy regularization.

    Optimistic actor:
        Learns an exploration policy using critic disagreement.

The optimistic actor adjusts the standard actor's Gaussian mean:

    mu_o = mu_c + delta_mu(s, mu_c)
    sigma_o = m * sigma_c

where m is the exploration standard-deviation multiplier.

The optimistic actor maximizes an upper-confidence value estimate:

    Q_upper = (Q1 + Q2)/2 + beta * |Q1 - Q2|/2

where beta controls the strength of optimism.

Key Component 4 — Adaptive Exploration Regularization
------------------------------------------------------
The optimistic actor is constrained to remain close to the
standard actor using KL divergence:

    L_optimistic = -E[Q_upper] + lambda * KL(pi_o || pi_c)

BRO automatically adapts:
    - beta: optimism strength.
    - lambda: KL regularization strength.

Both coefficients are adjusted to maintain a target KL divergence
between the standard and optimistic policies.

Key Component 5 — Network Resets
---------------------------------
BRO periodically resets:
    - Standard and optimistic actors.
    - Critic and target critic.
    - Optimizer states and adaptive coefficients.

The replay buffer is preserved, allowing newly initialized networks
to learn from previously collected experience.

BRO (Fast) uses fewer gradient updates and resets than full BRO.

Key Behaviour:
- Optimistic policy collects experience during training.
- Standard policy is used for evaluation and critic bootstrapping.
- Distributional critics provide return estimates and disagreement.
- Adaptive KL regularization controls exploration.
- Larger residual networks and resets support continued learning.

Algorithm Summary:
------------------
BRO = SAC
      + BroNet residual networks
      + twin quantile critics
      + optimistic exploration policy
      + adaptive KL regularization
      + weight decay and periodic resets.
"""

import logging
import math
import os
from typing import Any, Literal

import numpy as np
import torch

import cares_reinforcement_learning.algorithm.lossess as loss
from cares_reinforcement_learning.algorithm.configurations import BROConfig
from cares_reinforcement_learning.algorithm.policy.SAC import SAC
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.networks.BRO import Actor, Critic, OptimisticActor
from cares_reinforcement_learning.networks.bro_architecture import bro_weight_init
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation


class BRO(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        optimistic_actor_network: OptimisticActor,
        config: BROConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)

        self.bro_config = config
        self.num_quantiles = config.num_quantiles
        self.kappa = config.quantile_huber_kappa
        self.pessimism = config.pessimism
        self.std_multiplier = config.optimistic_std_multiplier
        self.kl_target = config.kl_target

        self.quantile_taus = torch.FloatTensor(
            [
                i / self.num_quantiles + 0.5 / self.num_quantiles
                for i in range(self.num_quantiles)
            ]
        ).to(device)

        self.optimistic_actor_net = optimistic_actor_network.to(device)

        self.optimistic_actor_net_optimiser = getattr(torch.optim, config.optimiser)(
            self.optimistic_actor_net.parameters(),
            lr=config.actor_lr,
            **config.actor_lr_params,
        )

        # The JAX Adjustment maps an unconstrained trainable parameter through
        # tanh into a bounded log-space range before exponentiating.
        self.adjustment_log_min = -10.0
        self.adjustment_log_max = 7.5
        self.raw_optimism = torch.nn.Parameter(
            torch.tensor(
                self._initial_adjustment(config.initial_optimism),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.raw_kl_weight = torch.nn.Parameter(
            torch.tensor(
                self._initial_adjustment(config.initial_kl_weight),
                device=self.device,
                dtype=torch.float32,
            )
        )
        self.optimism_optimiser = torch.optim.Adam(
            [self.raw_optimism], lr=config.optimism_lr, betas=(0.5, 0.999)
        )
        self.kl_optimiser = torch.optim.Adam(
            [self.raw_kl_weight], lr=config.kl_lr, betas=(0.5, 0.999)
        )

        self.reset_steps = set(config.reset_steps)
        self.last_reset_step = -1

    def _initial_adjustment(self, value: float) -> float:
        if value <= 0:
            raise ValueError("BRO adjustment initial values must be positive")
        normalized = (
            2.0
            * (math.log(value) - self.adjustment_log_min)
            / (self.adjustment_log_max - self.adjustment_log_min)
            - 1.0
        )
        if not -1.0 < normalized < 1.0:
            raise ValueError("BRO adjustment initial value outside supported range")
        return math.atanh(normalized)

    def _adjustment(self, raw_value: torch.Tensor) -> torch.Tensor:
        log_value = self.adjustment_log_min + (
            self.adjustment_log_max - self.adjustment_log_min
        ) * 0.5 * (1.0 + torch.tanh(raw_value))
        return torch.exp(log_value)

    @property
    def optimism(self) -> torch.Tensor:
        return self._adjustment(self.raw_optimism)

    @property
    def kl_weight(self) -> torch.Tensor:
        return self._adjustment(self.raw_kl_weight)

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        state = torch.as_tensor(
            observation.vector_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad():
            with fnc.evaluating(self.actor_net):
                if evaluation:
                    _, _, action = self.actor_net(state)
                else:
                    means, stds = self.actor_net.gaussian_parameters(state)
                    with fnc.evaluating(self.optimistic_actor_net):
                        action, _, _, _, _ = self.optimistic_actor_net(
                            state, means, stds, self.std_multiplier
                        )

        return ActionSample(action=action.squeeze(0).cpu().numpy(), source="policy")

    def _calculate_value(
        self, state: SARLObservation, action: np.ndarray
    ) -> float:  # type: ignore[override]
        state_tensor = torch.as_tensor(
            state.vector_state, dtype=torch.float32, device=self.device
        ).unsqueeze(0)
        action_tensor = torch.as_tensor(
            action, dtype=torch.float32, device=self.device
        ).unsqueeze(0)

        with torch.no_grad(), fnc.evaluating(self.critic_net):
            q1, q2 = self.critic_net(state_tensor, action_tensor)
            value = 0.5 * (q1 + q2).mean(dim=-1)
        return value.item()

    def _quantile_loss(
        self, predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor
    ) -> torch.Tensor:
        # [B, N_pred, N_target], matching the original BRO JAX implementation.
        td_errors = target.unsqueeze(1) - predicted.unsqueeze(2)
        huber = loss.calculate_huber_loss(
            td_errors, kappa=self.kappa, use_mean_reduction=False
        )
        quantile_weights = (
            self.quantile_taus.view(1, -1, 1) - (td_errors.detach() < 0).float()
        ).abs()

        # JAX: sum over predicted quantiles, then mean over the other axes.
        per_sample = (quantile_weights * huber / self.kappa).sum(dim=1).mean(dim=1)
        return (per_sample * weights.reshape(-1)).mean()

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        with torch.no_grad(), fnc.evaluating(self.actor_net):
            next_actions, next_log_pi, _ = self.actor_net(next_states)
            target_q1, target_q2 = self.target_critic_net(next_states, next_actions)
            target_q = (
                0.5 * (target_q1 + target_q2)
                - 0.5 * self.pessimism * (target_q1 - target_q2).abs()
            )
            q_target = rewards * self.reward_scale + self.gamma * (1 - dones) * (
                target_q - self.alpha * next_log_pi
            )

        q1, q2 = self.critic_net(states, actions)
        critic_loss_one = self._quantile_loss(q1, q_target, weights)
        critic_loss_two = self._quantile_loss(q2, q_target, weights)

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        with torch.no_grad():
            td1 = q1.mean(dim=-1, keepdim=True) - q_target.mean(dim=-1, keepdim=True)
            td2 = q2.mean(dim=-1, keepdim=True) - q_target.mean(dim=-1, keepdim=True)
            td_error = torch.maximum(td1.abs(), td2.abs())
            priorities = (
                td_error.clamp_min(self.min_priority)
                .pow(self.per_alpha)
                .flatten()
                .cpu()
                .numpy()
            )
            info = {
                "critic_loss_one": critic_loss_one.item(),
                "critic_loss_two": critic_loss_two.item(),
                "critic_loss_total": critic_loss_total.item(),
                "q1_mean": q1.mean().item(),
                "q2_mean": q2.mean().item(),
                "q_twin_gap_abs_mean": (q1 - q2).abs().mean().item(),
                "q_target_mean": q_target.mean().item(),
                "q_target_std": q_target.std(unbiased=False).item(),
                "td_abs_mean": td_error.mean().item(),
            }
        return info, priorities

    def _update_actor_alpha(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        # Exploitation actor: SAC entropy term and BRO risk-adjusted twin-Q.
        actions, log_pi, _ = self.actor_net(states)

        with fnc.evaluating(self.critic_net):
            q1, q2 = self.critic_net(states, actions)

        q = 0.5 * (q1 + q2) - 0.5 * self.pessimism * (q1 - q2).abs()

        actor_loss = (self.alpha * log_pi - q.mean(dim=-1, keepdim=True)).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # The optimistic actor uses the updated standard actor.
        with torch.no_grad():
            means, stds = self.actor_net.gaussian_parameters(states)

        optimistic_actions, _, _, optimistic_means, optimistic_stds = (
            self.optimistic_actor_net(states, means, stds, self.std_multiplier)
        )

        with fnc.evaluating(self.critic_net):
            q1_o, q2_o = self.critic_net(states, optimistic_actions)

        q_upper = (
            0.5 * (q1_o + q2_o) + 0.5 * self.optimism.detach() * (q1_o - q2_o).abs()
        )

        # Reference BRO KL calculation.
        std_o_unscaled = optimistic_stds / self.std_multiplier

        kl = (
            torch.log(stds / std_o_unscaled)
            + (std_o_unscaled.square() + (optimistic_means - means).square())
            / (2.0 * stds.square())
            - 0.5
        ).sum(dim=-1, keepdim=True)

        optimistic_loss = (
            -q_upper.mean(dim=-1, keepdim=True) + self.kl_weight.detach() * kl
        ).mean()

        self.optimistic_actor_net_optimiser.zero_grad()
        optimistic_loss.backward()
        self.optimistic_actor_net_optimiser.step()

        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        # JAX controls the average KL *per action dimension*.
        kl_gap = kl.detach().mean() / self.actor_net.num_actions - self.kl_target

        optimism_loss = (self.optimism - self.pessimism) * kl_gap
        self.optimism_optimiser.zero_grad()
        optimism_loss.backward()
        self.optimism_optimiser.step()

        regularizer_loss = -self.kl_weight * kl_gap
        self.kl_optimiser.zero_grad()
        regularizer_loss.backward()
        self.kl_optimiser.step()

        return {
            "actor_loss": actor_loss.detach().item(),
            "optimistic_actor_loss": optimistic_loss.detach().item(),
            "alpha_loss": alpha_loss.detach().item(),
            "alpha": self.alpha.item(),
            "log_pi_mean": log_pi.detach().mean().item(),
            "actor_q_mean": q.detach().mean().item(),
            "optimistic_q_mean": q_upper.detach().mean().item(),
            "optimistic_kl_mean": kl.detach().mean().item(),
            "optimistic_kl_per_action": (
                kl.detach().mean() / self.actor_net.num_actions
            ).item(),
            "optimism": self.optimism.detach().item(),
            "kl_weight": self.kl_weight.detach().item(),
            "optimism_loss": optimism_loss.detach().item(),
            "regularizer_loss": regularizer_loss.detach().item(),
        }

    def _reset_networks(self) -> None:
        """Full BRO reset: keep replay and environment-step schedule."""
        config = self.bro_config

        for network in (self.actor_net, self.optimistic_actor_net, self.critic_net):
            for module in network.modules():
                if isinstance(module, torch.nn.LayerNorm):
                    module.reset_parameters()
            network.apply(bro_weight_init)

        # The optimistic shift head has a special small initialization.
        torch.nn.init.orthogonal_(
            self.optimistic_actor_net.mean_shift.weight, gain=0.01
        )
        self.hard_update_params(self.critic_net, self.target_critic_net)

        optimiser_cls = getattr(torch.optim, config.optimiser)
        self.actor_net_optimiser = optimiser_cls(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.optimistic_actor_net_optimiser = optimiser_cls(
            self.optimistic_actor_net.parameters(),
            lr=config.actor_lr,
            **config.actor_lr_params,
        )
        self.critic_net_optimiser = optimiser_cls(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        with torch.no_grad():
            self.log_alpha.fill_(math.log(config.initial_alpha))
            self.raw_optimism.fill_(self._initial_adjustment(config.initial_optimism))
            self.raw_kl_weight.fill_(self._initial_adjustment(config.initial_kl_weight))

        self.log_alpha_optimizer = optimiser_cls(
            [self.log_alpha], lr=config.alpha_lr, **config.alpha_lr_params
        )
        self.optimism_optimiser = torch.optim.Adam(
            [self.raw_optimism], lr=config.optimism_lr, betas=(0.5, 0.999)
        )
        self.kl_optimiser = torch.optim.Adam(
            [self.raw_kl_weight], lr=config.kl_lr, betas=(0.5, 0.999)
        )

    def train(
        self, memory_buffer: SARLMemoryBuffer, episode_context: EpisodeContext
    ) -> dict[str, Any]:
        # Runner calls train() G times at the same environment step.
        step = episode_context.training_step + 1
        reset = step in self.reset_steps and step != self.last_reset_step
        if reset:
            self._reset_networks()
            self.last_reset_step = step

        info = super().train(memory_buffer, episode_context)

        info["network_reset"] = reset

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        os.makedirs(filepath, exist_ok=True)
        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "optimistic_actor": self.optimistic_actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "optimistic_actor_optimizer": self.optimistic_actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "log_alpha": self.log_alpha.detach().item(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
            "raw_optimism": self.raw_optimism.detach().item(),
            "raw_kl_weight": self.raw_kl_weight.detach().item(),
            "optimism_optimizer": self.optimism_optimiser.state_dict(),
            "kl_optimizer": self.kl_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
            "last_reset_step": self.last_reset_step,
        }
        torch.save(checkpoint, os.path.join(filepath, f"{filename}_checkpoint.pth"))
        logging.info("BRO models, optimizers, and training state saved...")

    def load_models(
        self,
        filepath: str,
        filename: str,
        load_mode: Literal["resume", "transfer"] = "resume",
    ) -> None:
        if load_mode not in ("resume", "transfer"):
            raise ValueError(f"Unknown load mode: {load_mode}")
        checkpoint = torch.load(
            os.path.join(filepath, f"{filename}_checkpoint.pth"), map_location="cpu"
        )
        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        if load_mode == "transfer":
            self.hard_update_params(self.critic_net, self.target_critic_net)
            logging.info("BRO standard actor and critic weights loaded for transfer...")
            return

        self.optimistic_actor_net.load_state_dict(checkpoint["optimistic_actor"])
        self.target_critic_net.load_state_dict(checkpoint["target_critic"])
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.optimistic_actor_net_optimiser.load_state_dict(
            checkpoint["optimistic_actor_optimizer"]
        )
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])

        with torch.no_grad():
            self.log_alpha.fill_(checkpoint["log_alpha"])
            self.raw_optimism.fill_(checkpoint["raw_optimism"])
            self.raw_kl_weight.fill_(checkpoint["raw_kl_weight"])

        self.log_alpha_optimizer.load_state_dict(checkpoint["log_alpha_optimizer"])
        self.optimism_optimiser.load_state_dict(checkpoint["optimism_optimizer"])
        self.kl_optimiser.load_state_dict(checkpoint["kl_optimizer"])
        self.learn_counter = checkpoint.get("learn_counter", 0)
        self.last_reset_step = checkpoint.get("last_reset_step", -1)
        logging.info("BRO models, optimizers, and training state loaded...")
