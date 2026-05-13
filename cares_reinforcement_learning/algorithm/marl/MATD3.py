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
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import MATD3Config
from cares_reinforcement_learning.algorithm.policy.TD3 import TD3
from cares_reinforcement_learning.algorithm.schedulers import ExponentialScheduler
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


@dataclass(frozen=True, slots=True)
class MATD3Batch:
    global_states: torch.Tensor
    next_global_states: torch.Tensor
    agent_states: dict[str, torch.Tensor]
    next_agent_states: dict[str, torch.Tensor]

    actions_by_agent: dict[str, torch.Tensor]
    rewards_by_agent: dict[str, torch.Tensor]
    dones_by_agent: dict[str, torch.Tensor]

    actions: torch.Tensor
    rewards: torch.Tensor
    dones: torch.Tensor
    next_actions: torch.Tensor
    next_actions_noisy: torch.Tensor
    joint_actions: torch.Tensor


class MATD3(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        learning_units: dict[str, TD3],
        all_agent_ids: list[str],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_to_agent_ids: dict[str, list[str]],
        config: MATD3Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.learning_units = learning_units

        # Shared Actor/Critic per team or separate Actor/Critic per agent
        self.sharing_mode = config.sharing_mode

        # All environment agent IDs and counts, e.g.
        # ["adversary_0", "adversary_1", "adversary_2", "agent_0"]
        self.all_agent_ids = all_agent_ids
        self.num_agents = len(all_agent_ids)

        # Maps env agent -> learning unit.
        # separate:
        #   adversary_0 -> adversary_0
        # team:
        #   adversary_0 -> adversary
        self.agent_id_to_learning_unit_id = agent_id_to_learning_unit_id

        # Maps learning unit -> env agents controlled by it.
        # separate:
        #   adversary_0 -> [adversary_0]
        # team:
        #   adversary -> [adversary_0, adversary_1, adversary_2]
        self.learning_unit_to_agent_ids = learning_unit_to_agent_ids

        self.controlled_agent_ids = [
            agent_id
            for controlled_agents in self.learning_unit_to_agent_ids.values()
            for agent_id in controlled_agents
        ]

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

        self.max_grad_norm = config.max_grad_norm

        self.learn_counter = 0

    def act(
        self, observation: MARLObservation, evaluation: bool = False
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_states = observation.agent_states
        avail_actions = observation.available_actions

        actions = {}

        for agent_id in self.controlled_agent_ids:
            learning_unit_id = self.agent_id_to_learning_unit_id[agent_id]
            learning_unit = self.learning_units[learning_unit_id]

            obs_i = agent_states[agent_id]
            avail_i = avail_actions[agent_id]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                available_actions=avail_i,
            )

            agent_sample = learning_unit.act(agent_observation, evaluation)
            actions[agent_id] = agent_sample.action

        return ActionSample(action=actions, source="policy")

    def _update_critic(
        self,
        learning_unit: TD3,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B, 1)
        next_global_states: torch.Tensor,
        next_actions_noisy: torch.Tensor,  # (B, N, act_dim) from target actors
        dones_i: torch.Tensor,
    ):
        info: dict[str, Any] = {}
        # --- Step 1: build next joint actions ---
        next_joint_actions_noisy = next_actions_noisy.reshape(
            next_actions_noisy.shape[0], -1
        )

        # --- Step 2: TD target ---
        with torch.no_grad():
            target_q_values_one, target_q_values_two = learning_unit.target_critic_net(
                next_global_states, next_joint_actions_noisy
            )
            target_q = torch.min(target_q_values_one, target_q_values_two)
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values_one, q_values_two = learning_unit.critic_net(
            global_states, joint_actions
        )

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)

        critic_loss_total = critic_loss_one + critic_loss_two

        learning_unit.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                learning_unit.critic_net.parameters(), self.max_grad_norm
            )

        learning_unit.critic_net_optimiser.step()

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

    def _build_actor_contribution(
        self,
        learning_unit: TD3,
        controlled_agent_id: str,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        replay_actions: torch.Tensor,  # (B, N, act_dim)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build the actor contribution for one controlled environment agent.
        """
        batch_size = global_states.shape[0]
        agent_index = self.all_agent_ids.index(controlled_agent_id)

        actions_all = replay_actions.clone()

        obs_i = obs_tensors[controlled_agent_id]
        actions_i = learning_unit.actor_net(obs_i)

        actions_all[:, agent_index, :] = actions_i

        joint_actions_flat = actions_all.reshape(batch_size, -1)

        actor_q_values, _ = learning_unit.critic_net(
            global_states,
            joint_actions_flat,
        )

        actor_objective = -actor_q_values.mean()

        return actions_i, actor_q_values, actor_objective

    def _update_actor(
        self,
        learning_unit: TD3,
        controlled_agent_ids: list[str],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        replay_actions: torch.Tensor,  # (B, N, act_dim)
    ) -> dict[str, Any]:
        """
        Unified MATD3 actor update.

        In separate mode `controlled_agent_ids` contains one agent ID.
        In team mode it contains every agent ID controlled by the shared learning unit.
        """
        info: dict[str, Any] = {}

        actions_all: list[torch.Tensor] = []
        actor_q_values_all: list[torch.Tensor] = []
        actor_objectives_all: list[torch.Tensor] = []

        for controlled_agent_id in controlled_agent_ids:
            actions_i, actor_q_values_i, actor_objective_i = (
                self._build_actor_contribution(
                    learning_unit=learning_unit,
                    controlled_agent_id=controlled_agent_id,
                    obs_tensors=obs_tensors,
                    global_states=global_states,
                    replay_actions=replay_actions,
                )
            )

            actions_all.append(actions_i)
            actor_q_values_all.append(actor_q_values_i)
            actor_objectives_all.append(actor_objective_i)

        actor_loss = torch.stack(actor_objectives_all).mean()

        actions_cat = torch.cat(actions_all, dim=0)
        actor_q_cat = torch.cat(actor_q_values_all, dim=0)

        dq_da_values = []
        for actor_q_values_i, actions_i in zip(actor_q_values_all, actions_all):
            dq_da = torch.autograd.grad(
                outputs=-actor_q_values_i.mean(),
                inputs=actions_i,
                retain_graph=True,
                create_graph=False,
            )[0]
            dq_da_values.append(dq_da.detach())

        dq_da_cat = torch.cat(dq_da_values, dim=0)

        learning_unit.actor_net_optimiser.zero_grad()
        actor_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                learning_unit.actor_net.parameters(),
                self.max_grad_norm,
            )

        learning_unit.actor_net_optimiser.step()

        with torch.no_grad():
            info["dq_da_abs_mean"] = dq_da_cat.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da_cat.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da_cat.norm(dim=1).quantile(0.95).item()

            info["pi_action_mean"] = actions_cat.mean().item()
            info["pi_action_std"] = actions_cat.std().item()
            info["pi_action_abs_mean"] = actions_cat.abs().mean().item()
            info["pi_action_saturation_frac"] = (
                (actions_cat.abs() > 0.95).float().mean().item()
            )

            info["actor_loss"] = actor_loss.item()
            info["actor_q_mean"] = actor_q_cat.mean().item()
            info["actor_q_std"] = actor_q_cat.std().item()

        return info

    def _sample_training_batch(
        self,
        memory_buffer: MARLMemoryBuffer,
    ) -> tuple[MATD3Batch, dict[str, Any]]:
        info: dict[str, Any] = {}

        sample_tensor, _ = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,
        )

        actions_by_agent = sample_tensor.action
        rewards_by_agent = sample_tensor.reward
        dones_by_agent = sample_tensor.done

        actions_tensor = torch.stack(
            [actions_by_agent[a] for a in self.all_agent_ids],
            dim=1,
        )

        rewards_tensor = torch.stack(
            [rewards_by_agent[a] for a in self.all_agent_ids],
            dim=1,
        )

        dones_tensor = torch.stack(
            [dones_by_agent[a] for a in self.all_agent_ids],
            dim=1,
        )

        next_actions: dict[str, torch.Tensor] = {}

        for agent_id in self.all_agent_ids:
            learning_unit_id = self.agent_id_to_learning_unit_id[agent_id]
            learning_unit = self.learning_units[learning_unit_id]

            obs_next_i = sample_tensor.next_observation.agent_states[agent_id]

            with torch.no_grad():
                with fnc.evaluating(learning_unit.target_actor_net):
                    next_actions[agent_id] = learning_unit.target_actor_net(obs_next_i)

        next_actions_tensor = torch.stack(
            [next_actions[a] for a in self.all_agent_ids],
            dim=1,
        )

        # TD3 target policy smoothing: only used for critic target actions
        target_noise = torch.randn_like(next_actions_tensor) * self.policy_noise
        target_noise = target_noise.clamp(
            -self.policy_noise_clip,
            self.policy_noise_clip,
        )

        next_actions_noisy = (next_actions_tensor + target_noise).clamp(-1.0, 1.0)

        joint_actions = actions_tensor.reshape(actions_tensor.shape[0], -1)

        with torch.no_grad():
            info["target_noise_abs_mean"] = target_noise.abs().mean().item()
            info["target_noise_clip_frac"] = (
                (target_noise.abs() >= self.policy_noise_clip).float().mean().item()
            )

            info["joint_action_abs_mean"] = actions_tensor.abs().mean().item()
            info["joint_action_std"] = actions_tensor.std(unbiased=False).item()
            info["action_saturation_frac"] = (
                (actions_tensor.abs() > 0.95).float().mean().item()
            )

            a_norm = actions_tensor / (actions_tensor.norm(dim=2, keepdim=True) + 1e-12)
            cos = torch.einsum("bna,bma->bnm", a_norm, a_norm)
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

        batch_data = MATD3Batch(
            global_states=sample_tensor.observation.global_state,
            next_global_states=sample_tensor.next_observation.global_state,
            agent_states=sample_tensor.observation.agent_states,
            next_agent_states=sample_tensor.next_observation.agent_states,
            actions_by_agent=actions_by_agent,
            rewards_by_agent=rewards_by_agent,
            dones_by_agent=dones_by_agent,
            actions=actions_tensor,
            rewards=rewards_tensor,
            dones=dones_tensor,
            next_actions=next_actions_tensor,
            next_actions_noisy=next_actions_noisy,
            joint_actions=joint_actions,
        )

        return batch_data, info

    def _train_separate(self, memory_buffer: MARLMemoryBuffer):
        info: dict[str, Any] = {}

        # ---------------------------------------------------------
        # Sample ONCE for all agents (recommended for TD3/SAC)
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across agent updates.
        # This preserves an unbiased estimator of each update while reducing sampling-induced variance and
        # keeping joint transitions consistent for centralized critics.
        # ---------------------------------------------------------
        samples, batch_info = self._sample_training_batch(memory_buffer)
        info |= batch_info

        global_states = samples.global_states
        next_global_states = samples.next_global_states
        agent_states = samples.agent_states
        rewards_by_agent = samples.rewards_by_agent
        dones_by_agent = samples.dones_by_agent
        actions = samples.actions
        next_actions_noisy = samples.next_actions_noisy
        joint_actions = samples.joint_actions

        # ---------------------------------------------------------
        # CRITIC UPDATES (every step)
        # ---------------------------------------------------------
        for learning_unit_id, learning_unit in self.learning_units.items():
            rewards_i = rewards_by_agent[learning_unit_id]
            dones_i = dones_by_agent[learning_unit_id]

            critic_info = self._update_critic(
                learning_unit=learning_unit,
                global_states=global_states,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_global_states,
                next_actions_noisy=next_actions_noisy,  # <-- noisy version
                dones_i=dones_i,
            )
            for key, value in critic_info.items():
                info[f"{learning_unit_id}_{key}"] = value

        # ---------------------------------------------------------
        # ACTOR + TARGET UPDATES (DELAYED — TD3)
        # ---------------------------------------------------------
        update_actor = self.learn_counter % self.policy_update_freq == 0
        if update_actor:
            for learning_unit_id, learning_unit in self.learning_units.items():
                actor_info = self._update_actor(
                    learning_unit=learning_unit,
                    controlled_agent_ids=[learning_unit_id],
                    obs_tensors=agent_states,
                    global_states=global_states,
                    replay_actions=actions,
                )
                for key, value in actor_info.items():
                    info[f"{learning_unit_id}_{key}"] = value

            # TD3: target networks updated on SAME cadence as actor
            for agent in self.learning_units.values():
                agent.update_target_networks()

        # --- Cross-agent diagnostics ---
        metrics = list(critic_info.keys())
        if update_actor:
            metrics += list(actor_info.keys())
        for metric in metrics:
            values = [
                info[f"{learning_unit_id}_{metric}"]
                for learning_unit_id in self.learning_units.keys()
            ]
            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

        return info

    def _train_team(
        self,
        memory_buffer: MARLMemoryBuffer,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        # ---------------------------------------------------------
        # Sample ONCE for all teams (recommended for MATD3)
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across team updates.
        # This preserves an unbiased estimator of each update while reducing sampling-induced variance and
        # keeping joint transitions consistent for centralized critics.
        samples, batch_info = self._sample_training_batch(memory_buffer)
        info |= batch_info

        global_states = samples.global_states
        next_global_states = samples.next_global_states
        agent_states = samples.agent_states
        rewards_by_agent = samples.rewards_by_agent
        dones_by_agent = samples.dones_by_agent
        actions = samples.actions
        next_actions_noisy = samples.next_actions_noisy
        joint_actions = samples.joint_actions

        # ---------------------------------------------------------
        # Update each TEAM
        # ---------------------------------------------------------
        for learning_unit_id, learning_unit in self.learning_units.items():
            controlled_agent_ids = self.learning_unit_to_agent_ids[learning_unit_id]

            # ---------------------------------------------------------
            # Aggregate rewards/dones across team
            # ---------------------------------------------------------
            learning_unit_rewards = torch.stack(
                [rewards_by_agent[a] for a in controlled_agent_ids],
                dim=1,
            ).mean(dim=1)

            learning_unit_dones = torch.stack(
                [dones_by_agent[a] for a in controlled_agent_ids],
                dim=1,
            ).amax(dim=1)

            # ---------------------------------------------------------
            # Team diagnostics
            # ---------------------------------------------------------
            with torch.no_grad():

                info[f"{learning_unit_id}_reward_mean"] = (
                    learning_unit_rewards.mean().item()
                )

                info[f"{learning_unit_id}_done_frac"] = (
                    learning_unit_dones.float().mean().item()
                )

            # ---------------------------------------------------------
            # Critic update
            # ---------------------------------------------------------
            critic_info = self._update_critic(
                learning_unit=learning_unit,
                global_states=global_states,
                joint_actions=joint_actions,
                rewards_i=learning_unit_rewards,
                next_global_states=next_global_states,
                next_actions_noisy=next_actions_noisy,
                dones_i=learning_unit_dones,
            )

            info.update({f"{learning_unit_id}_{k}": v for k, v in critic_info.items()})

            # ---------------------------------------------------------
            # ACTOR + TARGET UPDATES (DELAYED — TD3)
            # ---------------------------------------------------------
            update_actor = self.learn_counter % self.policy_update_freq == 0
            if update_actor:
                actor_info = self._update_actor(
                    learning_unit=learning_unit,
                    controlled_agent_ids=controlled_agent_ids,
                    obs_tensors=agent_states,
                    global_states=global_states,
                    replay_actions=actions,
                )

                info.update(
                    {f"{learning_unit_id}_{k}": v for k, v in actor_info.items()}
                )

                # TD3: target networks updated on SAME cadence as actor
                for learning_unit in self.learning_units.values():
                    learning_unit.update_target_networks()

        # ---------------------------------------------------------
        # Cross-team diagnostics
        # ---------------------------------------------------------
        metrics = list(critic_info.keys()) + list(actor_info.keys())

        for metric in metrics:

            values = [
                info[f"{learning_unit_id}_{metric}"]
                for learning_unit_id in self.learning_units.keys()
            ]

            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

        return info

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        self.learn_counter += 1
        info: dict[str, Any] = {}

        # Update per agent action noise for exploration (decayed over training)
        for agent_name, agent_network in self.learning_units.items():
            agent_network.action_noise = agent_network.action_noise_scheduler.get_value(
                episode_context.training_step
            )
            info[f"{agent_name}_current_action_noise"] = agent_network.action_noise

        # Update TD3 target policy smoothing noise (decayed over training)
        self.policy_noise = self.policy_noise_scheduler.get_value(
            episode_context.training_step
        )
        info["current_policy_noise"] = self.policy_noise

        if self.sharing_mode == "separate":
            info |= self._train_separate(memory_buffer)
        elif self.sharing_mode == "team":
            info |= self._train_team(memory_buffer)
        else:
            raise NotImplementedError(
                f"Sharing mode {self.sharing_mode} not implemented"
            )

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for learning_unit_id, learning_unit in self.learning_units.items():
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_{learning_unit_id}_checkpoint"
            learning_unit.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for learning_unit_id, learning_unit in self.learning_units.items():
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_{learning_unit_id}_checkpoint"
            learning_unit.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
