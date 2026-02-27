"""
MATD3 (Multi-Agent TD3) implementation notes
--------------------------------------------

This algorithm extends MADDPG with TD3 improvements:
twin critics, delayed policy updates, and target policy smoothing.

Replay sampling:
- A single minibatch is sampled per training iteration and reused across agents.
- This preserves unbiased updates while reducing variance and keeping joint
  transitions consistent for centralized critics.
- TD3 introduces explicit variance-reduction mechanisms (twin critics and
  target smoothing), making shared minibatch updates more stable than the
  original MADDPG per-agent sampling scheme.

Critic updates:
- Twin critics are trained using TD3-style targets with target policy smoothing.
- Noise is applied only to NEXT actions for critic targets to reduce
  overestimation bias.

Actor updates:
- Actors are deterministic and updated with a delayed frequency.
- When updating agent i, only agent i's action is replaced with the current
  actor output; other agents' actions come from the replay buffer.
- This mirrors MADDPG and avoids unnecessary coupling of agent updates.

No joint action resampling:
- TD3's stochasticity is confined to target policy smoothing.
- Resampling other agents' current actions is unnecessary and can increase
  variance without benefit for deterministic policy gradients.
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import MATD3Config
from cares_reinforcement_learning.algorithm.policy.TD3 import TD3
from cares_reinforcement_learning.algorithm.schedulers import ExponentialScheduler
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


class MATD3(MARLAlgorithm[list[np.ndarray]]):
    def __init__(
        self,
        agents: list[TD3],
        config: MATD3Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

        self.gamma = config.gamma
        self.tau = config.tau

        self.policy_update_freq = config.policy_update_freq

        # Policy noise
        self.policy_noise_clip = config.policy_noise_clip
        self.policy_noise_scheduler = ExponentialScheduler(
            start_value=config.policy_noise_start,
            end_value=config.policy_noise_end,
            decay_steps=config.policy_noise_decay,
        )
        self.policy_noise = self.policy_noise_scheduler.get_value(0)
        self.policy_noise_clip = config.policy_noise_clip

        self.max_grad_norm = config.max_grad_norm

        self.learn_counter = 0

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[list[np.ndarray]]:
        agent_states = observation.agent_states
        avail_actions = observation.avail_actions

        agent_ids = list(agent_states.keys())
        actions = []

        for i, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[i]  # consistent ordering in dict
            obs_i = agent_states[agent_name]
            avail_i = avail_actions[i]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                avail_actions=avail_i,
            )

            agent_sample = agent.act(agent_observation, evaluation)
            actions.append(agent_sample.action)

        return ActionSample(action=actions, source="policy")

    def _update_critic(
        self,
        agent: TD3,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B, 1)
        next_global_states: torch.Tensor,
        next_actions_tensor: torch.Tensor,  # (B, N, act_dim) from target actors
        dones_i: torch.Tensor,
    ):
        info: dict[str, Any] = {}
        # --- Step 1: build next joint actions ---
        next_joint_actions = next_actions_tensor.view(next_actions_tensor.size(0), -1)

        # --- Step 2: TD target ---
        with torch.no_grad():
            target_q_values_one, target_q_values_two = agent.target_critic_net(
                next_global_states, next_joint_actions
            )
            target_q = torch.min(target_q_values_one, target_q_values_two)
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values_one, q_values_two = agent.critic_net(global_states, joint_actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)

        critic_loss_total = critic_loss_one + critic_loss_two

        agent.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.critic_net.parameters(), self.max_grad_norm
            )

        agent.critic_net_optimiser.step()

        with torch.no_grad():
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

            # --- Losses (optimization progress; less diagnostic than TD/twin gaps) ---
            info["critic_loss_one"] = critic_loss_one.item()
            info["critic_loss_two"] = critic_loss_two.item()
            info["critic_loss_total"] = critic_loss_total.item()

        return info

    def _update_actor(
        self,
        agent: TD3,
        agent_index: int,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        actions_tensor: torch.Tensor,  # (B, N, act_dim)
    ):
        """
        Paper-faithful MATD3 actor update:
        - For j ≠ agent_index: use replay-buffer actions
        - For j == agent_index: use current actor output
        """
        info: dict[str, Any] = {}

        agent_ids = list(obs_tensors.keys())
        batch_size = global_states.shape[0]

        # ---------------------------------------------------------
        # Step 1: Start from replay-buffer joint actions
        #         actions_all: (B, N, A)
        # ---------------------------------------------------------
        actions_all = actions_tensor.clone()  # clone so we can overwrite

        # ---------------------------------------------------------
        # Step 2: Replace ONLY agent_i action with differentiable action
        # ---------------------------------------------------------
        obs_i = obs_tensors[agent_ids[agent_index]]  # (B, obs_dim_i)
        actions_i = agent.actor_net(obs_i)  # differentiable

        actions_all[:, agent_index, :] = actions_i  # keep others from buffer

        # ---------------------------------------------------------
        # Step 4: Compute actor loss: -Q_i(x, a_1,...,a_i,...,a_N)
        # ---------------------------------------------------------
        joint_actions_flat = actions_all.reshape(batch_size, -1)
        actor_q_values, _ = agent.critic_net(global_states, joint_actions_flat)

        actor_loss = -actor_q_values.mean()

        # ---------------------------------------------------------
        # Deterministic Policy Gradient Strength (∇a Q(s,a))
        # ---------------------------------------------------------
        # Measures how steep the critic surface is w.r.t. actions.
        # ~0 early  -> critic flat, actor receives no learning signal.
        # Very large -> critic overly sharp, can cause unstable actor updates.
        dq_da = torch.autograd.grad(
            outputs=actor_loss,
            inputs=actions_i,
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

        # ---------------------------------------------------------
        # Step 5: Backprop
        # ---------------------------------------------------------
        agent.actor_net_optimiser.zero_grad()
        actor_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.actor_net.parameters(), self.max_grad_norm
            )

        agent.actor_net_optimiser.step()

        with torch.no_grad():
            # Policy Action Health (tanh policies in [-1, 1])
            # pi_action_saturation_frac:
            # High values (>0.8 early) often mean the actor is slamming bounds,
            # reducing effective gradient flow through tanh.
            info["pi_action_mean"] = actions_i.mean().item()
            info["pi_action_std"] = actions_i.std().item()
            info["pi_action_abs_mean"] = actions_i.abs().mean().item()
            info["pi_action_saturation_frac"] = (
                (actions_i.abs() > 0.95).float().mean().item()
            )

            # actor_q_mean should generally increase over training.
            # actor_q_std large + unstable may indicate critic inconsistency.
            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_values.mean().item()
            info["actor_q_std"] = actor_q_values.std().item()

        return info

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        self.learn_counter += 1

        info: dict[str, Any] = {}

        # Update per agent action noise for exploration (decayed over training)
        for i, current_agent in enumerate(self.agent_networks):
            current_agent.action_noise = current_agent.action_noise_scheduler.get_value(
                episode_context.training_step
            )
            info[f"agent_{i}_current_action_noise"] = current_agent.action_noise

        # Update TD3 target policy smoothing noise (decayed over training)
        self.policy_noise = self.policy_noise_scheduler.get_value(
            episode_context.training_step
        )
        info["current_policy_noise"] = self.policy_noise

        # ---------------------------------------------------------
        # Sample ONCE for all agents (recommended for TD3/SAC)
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across agent updates.
        # This preserves an unbiased estimator of each update while reducing sampling-induced variance and
        # keeping joint transitions consistent for centralized critics.
        # ---------------------------------------------------------
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,
            _,
            indices,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,
        )

        batch_size = len(indices)

        global_states = observation_tensor.global_state_tensor
        next_global_states = next_observation_tensor.global_state_tensor

        agent_states = observation_tensor.agent_states_tensor
        next_agent_states = next_observation_tensor.agent_states_tensor

        agent_ids = list(agent_states.keys())

        # Flatten replay-buffer joint actions (used for critic current Q)
        joint_actions = actions_tensor.reshape(batch_size, -1)

        # ---------------------------------------------------------
        # Build NEXT actions using TARGET actors (clean)
        # ---------------------------------------------------------
        next_actions = []
        for agent, agent_id in zip(self.agent_networks, agent_ids):
            obs_next = next_agent_states[agent_id]
            with hlp.evaluating(agent.target_actor_net):
                next_actions.append(agent.target_actor_net(obs_next))

        # (B, N, act_dim)
        next_actions_tensor = torch.stack(next_actions, dim=1)

        # ---------------------------------------------------------
        # TD3 TARGET POLICY SMOOTHING (ONCE)
        # ---------------------------------------------------------
        # This affects ONLY critic targets
        target_noise = torch.randn_like(next_actions_tensor) * self.policy_noise
        target_noise = target_noise.clamp(
            -self.policy_noise_clip, self.policy_noise_clip
        )

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

        # assumes tanh policy -> [-1, 1]
        next_actions_noisy = (next_actions_tensor + target_noise).clamp(-1.0, 1.0)

        # ---------------------------------------------------------
        # CRITIC UPDATES (every step)
        # ---------------------------------------------------------
        for agent_index, agent in enumerate(self.agent_networks):
            rewards_i = rewards_tensor[:, agent_index]
            dones_i = dones_tensor[:, agent_index]

            critic_info = self._update_critic(
                agent=agent,
                global_states=global_states,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_global_states,
                next_actions_tensor=next_actions_noisy,  # <-- noisy version
                dones_i=dones_i,
            )
            for key, value in critic_info.items():
                info[f"agent_{agent_index}_{key}"] = value

        # ---------------------------------------------------------
        # ACTOR + TARGET UPDATES (DELAYED — TD3)
        # ---------------------------------------------------------
        update_actor = self.learn_counter % self.policy_update_freq == 0
        if update_actor:
            for agent_index, agent in enumerate(self.agent_networks):
                actor_info = self._update_actor(
                    agent=agent,
                    agent_index=agent_index,
                    obs_tensors=agent_states,
                    global_states=global_states,
                    actions_tensor=actions_tensor,
                )
                for key, value in actor_info.items():
                    info[f"agent_{agent_index}_{key}"] = value

            # TD3: target networks updated on SAME cadence as actor
            for agent in self.agent_networks:
                agent.update_target_networks()

        with torch.no_grad():
            # --- Joint action distribution (from replay) ---
            # Detects action collapse / saturation / scaling issues across agents
            info["joint_action_abs_mean"] = actions_tensor.abs().mean().item()
            info["joint_action_std"] = actions_tensor.std(unbiased=False).item()
            info["action_saturation_frac"] = (
                (actions_tensor.abs() > 0.95).float().mean().item()
            )

            # --- Coordination proxy on replay actions ---
            # Cos similarity between agents' action vectors per sample (mean over pairs)
            a = actions_tensor  # (B,N,A)
            a_norm = a / (a.norm(dim=2, keepdim=True) + 1e-12)
            cos = torch.einsum("bna,bma->bnm", a_norm, a_norm)  # (B,N,N)
            n = cos.shape[1]
            mask = ~torch.eye(n, device=cos.device, dtype=torch.bool)
            info["replay_action_cos_mean"] = cos[:, mask].mean().item()
            info["replay_action_cos_p95"] = cos[:, mask].quantile(0.95).item()

            info["next_action_abs_mean_all_agents"] = (
                next_actions_tensor.abs().mean().item()
            )
            info["next_action_std_all_agents"] = next_actions_tensor.std(
                unbiased=False
            ).item()
            info["next_action_saturation_frac_all_agents"] = (
                (next_actions_tensor.abs() > 0.95).float().mean().item()
            )

        # --- Cross-agent diagnostics ---
        metrics = list(critic_info.keys())
        if update_actor:
            metrics += list(actor_info.keys())
        for metric in metrics:
            values = [info[f"agent_{i}_{metric}"] for i in range(self.num_agents)]
            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
