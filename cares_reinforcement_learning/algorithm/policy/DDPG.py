"""
DDPG (Deep Deterministic Policy Gradient)
------------------------------------------

Original Paper: https://arxiv.org/pdf/1509.02971v5.pdf

DDPG is an off-policy actor-critic algorithm for continuous
action spaces. It combines deterministic policy gradients
with deep function approximation and target networks.

Architecture:
- Actor: deterministic policy π(s) → a
- Critic: Q(s, a) estimating expected return
- Target actor and critic networks for stability

Data / Training (off-policy):
- Transitions are sampled uniformly from a replay buffer.
- Uses one critic (unlike TD3 which uses two).
- Target networks are softly updated using:
      θ_target ← τ θ + (1 - τ) θ_target

Critic update:
- Target action from target actor:
      a' = π_target(s')
- Bellman target:
      y = r + γ (1 - done) Q_target(s', a')
- Minimize MSE:
      L = (Q(s, a) - y)²

Actor update:
- Deterministic policy gradient:
      ∇θ J ≈ E[ ∇a Q(s, a) ∇θ π(s) ]
- Implemented by maximizing Q(s, π(s))
  (minimizing -Q.mean()).

Exploration:
- Exploration noise is typically added externally to actions.
- This implementation leaves noise handling to higher-level logic.

Rationale:
- Enables stable continuous control via replay buffers and
  target networks.
- Deterministic policy reduces variance compared to
  stochastic policy gradients.

DDPG = Deterministic Actor-Critic + Replay Buffer + Target Networks.
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import SARLAlgorithm
from cares_reinforcement_learning.memory.memory_buffer import SARLMemoryBuffer
from cares_reinforcement_learning.networks.DDPG import Actor, Critic
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    SARLObservationTensors,
)
from cares_reinforcement_learning.util.configurations import DDPGConfig
from cares_reinforcement_learning.util.helpers import ExponentialScheduler


class DDPG(SARLAlgorithm[np.ndarray]):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: DDPGConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = config.gamma
        self.tau = config.tau

        # Action noise
        self.action_noise_scheduler = ExponentialScheduler(
            start_value=config.action_noise_start,
            end_value=config.action_noise_end,
            decay_steps=config.action_noise_decay,
        )
        self.action_noise = self.action_noise_scheduler.get_value(0)

        self.action_num = self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

        self.learn_counter = 0

    def act(
        self, observation: SARLObservation, evaluation: bool = False
    ) -> ActionSample[np.ndarray]:
        self.actor_net.eval()

        state = observation.vector_state

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the DDPG too, add noise to the action
                noise = np.random.normal(
                    0, scale=self.action_noise, size=self.action_num
                ).astype(np.float32)
                action = action + noise
                action = np.clip(action, -1, 1)

        self.actor_net.train()

        return ActionSample(action=action, source="policy")

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        with torch.no_grad():
            self.target_actor_net.eval()
            next_actions = self.target_actor_net(next_states)
            self.target_actor_net.train()

            target_q_values = self.target_critic_net(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values = self.critic_net(states, actions)

        critic_loss = F.mse_loss(q_values, q_target)
        self.critic_net_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net_optimiser.step()

        with torch.no_grad():
            td = q_values - q_target

            # --- Q statistics ---
            info["q_mean"] = q_values.mean().item()
            info["q_std"] = q_values.std().item()

            # --- Bellman target scale (reward scaling / discount sanity) ---
            # If q_target drifts upward without reward improvement, suspect reward_scale, gamma, or instability.
            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std().item()

            # --- TD error diagnostics (Bellman fit quality) ---
            # td_abs_mean down over time is healthy; persistent growth/spikes often indicate critic instability.
            info["td_mean"] = td.mean().item()
            info["td_std"] = td.std().item()
            info["td_abs_mean"] = td.abs().mean().item()

            # --- Losses (optimization progress ---
            info["critic_loss"] = critic_loss.item()

        return info

    def _update_actor(self, states: torch.Tensor) -> dict[str, Any]:
        info: dict[str, Any] = {}

        self.critic_net.eval()
        actions = self.actor_net(states)
        actor_q_values = self.critic_net(states, actions)
        self.critic_net.train()

        actor_loss = -actor_q_values.mean()

        # ---------------------------------------------------------
        # Deterministic Policy Gradient Strength (∇a Q(s,a))
        # ---------------------------------------------------------
        # Measures how steep the critic surface is w.r.t. actions.
        # ~0 early  -> critic flat, actor receives no learning signal.
        # Very large -> critic overly sharp, can cause unstable actor updates.
        dq_da = torch.autograd.grad(
            outputs=-actor_q_values.mean(),  # NOTE: uses Q-term only, excludes regularizers
            inputs=actions,
            retain_graph=True,  # needed because we will backward (actor_loss) next
            create_graph=False,  # diagnostic only
            allow_unused=False,
        )[0]
        with torch.no_grad():
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

            # actor_q_mean should generally increase over training.
            # actor_q_std large + unstable may indicate critic inconsistency.
            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_values.mean().item()
            info["actor_q_std"] = actor_q_values.std().item()

        return info

    def update_from_batch(
        self,
        episode_context: EpisodeContext,
        observation_tensor: SARLObservationTensors,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_observation_tensor: SARLObservationTensors,
        dones_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        self.learn_counter += 1

        info: dict[str, Any] = {}

        self.action_noise = self.action_noise_scheduler.get_value(
            episode_context.training_step
        )

        # Update Critic
        critic_info = self._update_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
        )
        info |= critic_info

        # Update Actor
        actor_info = self._update_actor(observation_tensor.vector_state_tensor)
        info |= actor_info

        self.update_target_networks()

        return info

    def train(
        self,
        memory_buffer: SARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        # Use the helper to sample and prepare tensors in one step
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,
            _,
            _,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # DDPG uses uniform sampling
        )

        info = self.update_from_batch(
            episode_context=episode_context,
            observation_tensor=observation_tensor,
            actions_tensor=actions_tensor,
            rewards_tensor=rewards_tensor,
            next_observation_tensor=next_observation_tensor,
            dones_tensor=dones_tensor,
        )

        return info

    def update_target_networks(self) -> None:
        hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        logging.info("models and optimisers have been loaded...")
