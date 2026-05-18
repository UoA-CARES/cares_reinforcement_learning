"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation notes
--------------------------------------------------------------------

Original Paper: https://arxiv.org/abs/2103.01955

MAPPO implementation overview
-----------------------------

MAPPO extends PPO to the multi-agent setting using:

    - centralized value critics,
    - decentralized stochastic actors,
    - generalized advantage estimation (GAE),
    - clipped PPO policy optimisation,
    - and optional team-based parameter sharing.

This implementation additionally supports several multi-agent coordination and
parameter-sharing variants through the configuration option:

    parameter_sharing_scope:
        "individual"   -> one PPO learning unit per environment agent
        "team_critic" -> one shared critic per team, separate actor per agent
        "team_all"    -> one shared PPO learning unit per team of agents

Terminology
-----------

Environment agent:
    An agent that exists in the environment, e.g.
        adversary_0, adversary_1, agent_0

Learning unit:
    A trainable PPO bundle containing:
        actor, critic, optimisers, entropy scheduling, and PPO update logic.

    In individual mode:
        each environment agent has its own learning unit.

    In team_critic mode:
        actors remain per-agent while critics are shared across the team.

    In team_all mode:
        all agents in the same team share one PPO learning unit.

Controlled agents:
    The environment agents assigned to a learning unit.

Centralized training, decentralized execution
---------------------------------------------

MAPPO follows the CTDE (centralized training, decentralized execution)
paradigm.

Training:
    critics receive:
        - the global state,
        - and centralized rollout information.

Execution:
    actors receive only the local observation of the environment agent they are
    producing an action for.

Even in team_all mode, actions are still produced independently per environment
agent. The shared actor is simply reused across multiple agents belonging to
the same team.

PPO extensions
--------------

Clipped PPO objective:
    MAPPO uses the PPO clipped policy objective:

        L_clip = min(
            r_t * A_t,
            clip(r_t, 1 - eps, 1 + eps) * A_t
        )

    where:

        r_t = pi(a_t|s_t) / pi_old(a_t|s_t)

    This constrains policy updates and improves training stability.

Generalized Advantage Estimation (GAE):
    Advantages are computed using GAE:

        A_t = delta_t + (gamma * lambda) * delta_{t+1} + ...

    This reduces variance while preserving low-bias policy gradients.

Entropy regularization:
    MAPPO includes entropy regularization during actor optimisation:

        L_actor = L_clip + entropy_coef * H(pi)

    This encourages exploration and prevents premature policy collapse.

Shared rollout sampling
-----------------------

A single rollout batch is collected and reused across all actor and critic
updates during that PPO iteration.

This keeps all learning units training against the same trajectory distribution
while reducing variance between updates and improving consistency for
centralized critics.

Critic update
-------------

For each critic learning unit:

    1. Compute centralized value estimates from the global state.
    2. Compute returns and advantages using GAE.
    3. Regress the critic against the computed returns:

            V(s) -> R_t

In individual mode:
    each critic optimizes only its controlled agent's returns.

In team_critic and team_all modes:
    critics optimize centralized team value estimates.

Actor update
------------

Actors are updated using the PPO clipped objective.

individual:
    each actor performs its own PPO update independently.

team_critic:
    each actor still performs an independent PPO update,
    while advantages are produced by a shared team critic.

team_all:
    all agents within the same team contribute to one shared PPO actor update.

For shared actor updates:

    1. PPO losses are computed for all controlled agents.
    2. The losses are averaged together.
    3. One shared optimizer step is applied.

This produces a coupled shared-policy update under the current joint trajectory
distribution.

KL early stopping
-----------------

MAPPO optionally supports PPO-style KL early stopping using:

    target_kl

KL is measured AFTER the optimizer step, matching common PPO practice.

individual / team_critic:
    KL stopping applies per actor.

team_all:
    KL stopping applies to the entire shared actor group.

If any controlled agent exceeds the KL threshold after the shared update,
future updates for that shared actor are skipped for the remainder of the PPO
iteration.

Practical notes
---------------

For cooperative environments such as simple_spread:

    team_critic:
        often a strong default because:
            - critics learn a centralized cooperative objective,
            - while actors remain agent-specific and relatively stable.

        This typically provides a good balance between:
            - coordination,
            - stability,
            - and optimization simplicity.

    team_all:
        can produce stronger coordinated behaviour because:
            - both actor and critic are shared across the team,
            - and PPO updates optimise a coupled team objective.

        This configuration is often most effective for homogeneous agents.

    individual:
        closest to fully separate PPO learning units.

        Each agent learns:
            - its own actor,
            - and its own critic.

        This provides the strongest specialization but may learn cooperative
        coordination more slowly in fully cooperative environments.

Compared to deterministic methods such as MADDPG/MATD3, MAPPO is typically:
    - more stable,
    - more robust to non-stationarity,
    - less sensitive to critic overestimation,
    - but potentially less sample efficient due to on-policy training.
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
from cares_reinforcement_learning.algorithm.configurations import MAPPOConfig
from cares_reinforcement_learning.algorithm.policy.PPO import PPO
from cares_reinforcement_learning.algorithm.schedulers import ExponentialScheduler
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


@dataclass(frozen=True, slots=True)
class MAPPOBatch:
    batch_size: int

    global_states: torch.Tensor
    next_global_states: torch.Tensor
    agent_states: dict[str, torch.Tensor]

    actions_by_agent: dict[str, torch.Tensor]
    rewards_by_agent: dict[str, torch.Tensor]
    dones_by_agent: dict[str, torch.Tensor]
    old_log_probs_by_agent: dict[str, torch.Tensor]


class MAPPO(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        learning_units: dict[str, PPO],
        all_agent_ids: list[str],
        env_teams: dict[str, list[str]],
        agent_id_to_actor_id: dict[str, str],
        actor_id_to_agent_ids: dict[str, list[str]],
        agent_id_to_critic_id: dict[str, str],
        critic_id_to_agent_ids: dict[str, list[str]],
        config: MAPPOConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        # Physical trainable containers.
        #
        # Each learning unit is a PPO bundle containing:
        #   - actor
        #   - centralized value critic
        #   - actor optimiser
        #   - critic optimiser
        #
        # IMPORTANT:
        #   Actor ownership and critic ownership are now decoupled.
        #
        #   A learning unit may therefore be used:
        #       - only as an actor provider,
        #       - only as a critic provider,
        #       - or as both.
        #
        # Example:
        #
        #   team_critic:
        #
        #       actor units:
        #           adversary_0
        #           adversary_1
        #           adversary_2
        #
        #       critic unit:
        #           adversary
        #
        #       Here:
        #           adversary_0 actor updates use:
        #               actor from learning_units["adversary_0"]
        #               critic from learning_units["adversary"]
        #
        # This separation allows:
        #   - per-agent actors
        #   - shared team critics
        #   - shared team actors
        #
        # while still reusing the underlying DDPG container abstraction.
        self.learning_units = learning_units

        # Shared Actor/Critic per team or individual Actor/Critic per agent
        self.parameter_sharing_scope = config.parameter_sharing_scope

        # All environment agent IDs and counts, e.g.
        # ["adversary_0", "adversary_1", "adversary_2", "agent_0"]
        self.all_agent_ids = all_agent_ids
        self.num_agents = len(all_agent_ids)

        self.env_teams = env_teams

        # Maps env agent -> actor learning unit.
        #
        # Example environment:
        #   adversary_0, adversary_1, adversary_2, agent_0
        #
        # individual or team_critic:
        #   {
        #       "adversary_0": "adversary_0",
        #       "adversary_1": "adversary_1",
        #       "adversary_2": "adversary_2",
        #       "agent_0": "agent_0",
        #   }
        #
        # team_all:
        #   {
        #       "adversary_0": "adversary",
        #       "adversary_1": "adversary",
        #       "adversary_2": "adversary",
        #       "agent_0": "agent",
        #   }
        #
        # Used for:
        #   - action selection
        #   - actor updates
        self.agent_id_to_actor_id = agent_id_to_actor_id

        # Maps actor learning unit -> env agents controlled by that actor.
        #
        # Example environment:
        #   adversary_0, adversary_1, adversary_2, agent_0
        #
        # individual or team_critic:
        #   {
        #       "adversary_0": ["adversary_0"],
        #       "adversary_1": ["adversary_1"],
        #       "adversary_2": ["adversary_2"],
        #       "agent_0": ["agent_0"],
        #   }
        #
        # team_all:
        #   {
        #       "adversary": [
        #           "adversary_0",
        #           "adversary_1",
        #           "adversary_2",
        #       ],
        #       "agent": ["agent_0"],
        #   }
        #
        # Used for:
        #   - actor update grouping
        #   - determining which agents' actions are replaced during
        #     deterministic policy gradient updates
        self.actor_id_to_agent_ids = actor_id_to_agent_ids

        # Maps env agent -> critic learning unit.
        #
        # Example environment:
        #   adversary_0, adversary_1, adversary_2, agent_0
        #
        # individual:
        #   {
        #       "adversary_0": "adversary_0",
        #       "adversary_1": "adversary_1",
        #       "adversary_2": "adversary_2",
        #       "agent_0": "agent_0",
        #   }
        #
        # team_critic or team_all:
        #   {
        #       "adversary_0": "adversary",
        #       "adversary_1": "adversary",
        #       "adversary_2": "adversary",
        #       "agent_0": "agent",
        #   }
        #
        # Used for:
        #   - critic updates
        #   - reward/done aggregation
        #   - critic evaluation during actor optimisation
        self.agent_id_to_critic_id = agent_id_to_critic_id

        # Maps critic learning unit -> env agents assigned to that critic.
        #
        # Example environment:
        #   adversary_0, adversary_1, adversary_2, agent_0
        #
        # individual:
        #   {
        #       "adversary_0": ["adversary_0"],
        #       "adversary_1": ["adversary_1"],
        #       "adversary_2": ["adversary_2"],
        #       "agent_0": ["agent_0"],
        #   }
        #
        # team_critic or team_all:
        #   {
        #       "adversary": [
        #           "adversary_0",
        #           "adversary_1",
        #           "adversary_2",
        #       ],
        #       "agent": ["agent_0"],
        #   }
        #
        # Used for:
        #   - team reward aggregation
        #   - team done aggregation
        #   - critic update grouping
        self.critic_id_to_agent_ids = critic_id_to_agent_ids

        self.controlled_agent_ids = [
            agent_id
            for controlled_agents in self.actor_id_to_agent_ids.values()
            for agent_id in controlled_agents
        ]

        self.minibatch_size = config.minibatch_size
        self.updates_per_iteration = config.updates_per_iteration

        self.entropy_scheduler = ExponentialScheduler(
            start_value=config.entropy_start,
            end_value=config.entropy_end,
            decay_steps=config.entropy_decay,
        )
        # initial entropy coefficient
        self.entropy_coef = self.entropy_scheduler.get_value(0)

        self.target_kl = config.target_kl

        self.max_grad_norm = config.max_grad_norm

        self.gae_lambda = config.gae_lambda

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_states = observation.agent_states
        available_actions = observation.available_actions

        actions = {}
        log_probs = {}

        for agent_id in self.controlled_agent_ids:
            learning_unit_id = self.agent_id_to_actor_id[agent_id]
            learning_unit = self.learning_units[learning_unit_id]

            obs_i = agent_states[agent_id]
            avail_i = available_actions[agent_id]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                available_actions=avail_i,
            )

            agent_sample = learning_unit.act(
                agent_observation, evaluation, calculate_value=False
            )
            actions[agent_id] = agent_sample.action
            log_probs[agent_id] = agent_sample.extras["log_prob"]

        return ActionSample(
            action=actions, source="policy", extras={"log_prob": log_probs}
        )

    def _update_critic_minibatch(
        self,
        critic_unit: PPO,
        critic_agent_ids: list[str],
        mb: torch.Tensor,
        global_states: torch.Tensor,
        returns_all: torch.Tensor,
    ) -> dict[str, Any]:
        critic_agent_indices = [
            self.all_agent_ids.index(agent_id) for agent_id in critic_agent_ids
        ]

        v_pred = critic_unit.critic_net(global_states[mb])
        v_pred = v_pred.view(len(mb), len(critic_agent_ids))

        v_targ = returns_all[mb][:, critic_agent_indices]

        critic_loss = F.mse_loss(v_pred, v_targ)

        critic_unit.critic_net_optimiser.zero_grad()
        critic_loss.backward()

        torch.nn.utils.clip_grad_norm_(
            critic_unit.critic_net.parameters(),
            self.max_grad_norm,
        )

        critic_unit.critic_net_optimiser.step()

        with torch.no_grad():
            value_error = v_pred - v_targ

            info = {
                "critic_loss": critic_loss.item(),
                "value_error_mean": value_error.mean().item(),
                "value_error_abs_mean": value_error.abs().mean().item(),
                "value_mean": v_pred.mean().item(),
                "return_mean": v_targ.mean().item(),
            }

        return info

    def _evaluate_actor_minibatch(
        self,
        actor_unit: PPO,
        agent_id: str,
        mb: torch.Tensor,
        agent_states: dict[str, torch.Tensor],
        actions_by_agent: dict[str, torch.Tensor],
        old_log_probs_by_agent: dict[str, torch.Tensor],
        advantages_all: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        agent_index = self.all_agent_ids.index(agent_id)

        states_mb = agent_states[agent_id][mb]
        actions_mb = actions_by_agent[agent_id][mb]
        old_logp_mb = old_log_probs_by_agent[agent_id][mb]
        advantages_mb = advantages_all[mb, agent_index]

        logp, entropy, _ = actor_unit._evaluate_actions(
            states=states_mb,
            actions=actions_mb,
        )

        log_ratio = logp - old_logp_mb
        ratio = torch.exp(log_ratio)

        unclipped_objective = ratio * advantages_mb

        clipped_ratio = torch.clamp(
            ratio,
            1.0 - actor_unit.eps_clip,
            1.0 + actor_unit.eps_clip,
        )
        clipped_objective = clipped_ratio * advantages_mb

        policy_objective = torch.min(
            unclipped_objective,
            clipped_objective,
        )

        policy_loss = -policy_objective.mean()
        entropy_mean = entropy.mean()

        actor_loss = policy_loss - self.entropy_coef * entropy_mean

        with torch.no_grad():
            clip_frac = (
                (
                    (ratio > 1.0 + actor_unit.eps_clip)
                    | (ratio < 1.0 - actor_unit.eps_clip)
                )
                .float()
                .mean()
            )

            actor_info = {
                "actor_loss": float(actor_loss.item()),
                "policy_loss": float(policy_loss.item()),
                "entropy": float(entropy_mean.item()),
                "clip_frac": float(clip_frac.item()),
                "ratio_mean": float(ratio.mean().item()),
                "ratio_std": float(ratio.std(unbiased=False).item()),
                "log_ratio_mean": float(log_ratio.mean().item()),
                "log_ratio_std": float(log_ratio.std(unbiased=False).item()),
                "log_ratio_max_abs": float(log_ratio.abs().max().item()),
            }

        return actor_loss, actor_info

    def _compute_post_update_kl(
        self,
        actor_unit: PPO,
        agent_id: str,
        mb: torch.Tensor,
        agent_states: dict[str, torch.Tensor],
        actions_by_agent: dict[str, torch.Tensor],
        old_log_probs_by_agent: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        states_mb = agent_states[agent_id][mb]
        actions_mb = actions_by_agent[agent_id][mb]
        old_logp_mb = old_log_probs_by_agent[agent_id][mb]

        with torch.no_grad():
            new_logp, _, u_mb = actor_unit._evaluate_actions(
                states=states_mb,
                actions=actions_mb,
            )

            new_log_ratio = new_logp - old_logp_mb
            new_ratio = torch.exp(new_log_ratio)

            approx_kl = (new_ratio - 1.0 - new_log_ratio).mean()

            clip_frac = (
                (
                    (new_ratio > 1.0 + actor_unit.eps_clip)
                    | (new_ratio < 1.0 - actor_unit.eps_clip)
                )
                .float()
                .mean()
            )

            sat_rate = (actions_mb.abs() > 0.99).float().mean()
            u_abs_mean = u_mb.abs().mean()
            u_abs_max = u_mb.abs().max()

        return {
            "approx_kl": float(approx_kl.item()),
            "post_clip_frac": float(clip_frac.item()),
            "post_ratio_mean": float(new_ratio.mean().item()),
            "post_ratio_std": float(new_ratio.std(unbiased=False).item()),
            "action_sat_rate": float(sat_rate.item()),
            "u_abs_mean": float(u_abs_mean.item()),
            "u_abs_max": float(u_abs_max.item()),
            "post_log_ratio_mean": float(new_log_ratio.mean().item()),
            "post_log_ratio_std": float(new_log_ratio.std(unbiased=False).item()),
            "post_log_ratio_max_abs": float(new_log_ratio.abs().max().item()),
        }

    def _update_actor_minibatch(
        self,
        actor_unit: PPO,
        agent_ids: list[str],
        mb: torch.Tensor,
        agent_states: dict[str, torch.Tensor],
        actions_by_agent: dict[str, torch.Tensor],
        old_log_probs_by_agent: dict[str, torch.Tensor],
        advantages_all: torch.Tensor,
        agent_kl_early_stop: dict[str, bool],
    ) -> tuple[dict[str, bool], dict[str, dict[str, Any]]]:
        actor_losses: list[torch.Tensor] = []
        actor_info_by_agent: dict[str, dict[str, Any]] = {}

        active_agent_ids = [
            agent_id for agent_id in agent_ids if not agent_kl_early_stop[agent_id]
        ]

        if not active_agent_ids:
            return (
                {agent_id: True for agent_id in agent_ids},
                actor_info_by_agent,
            )

        for agent_id in active_agent_ids:
            actor_loss_i, actor_info_i = self._evaluate_actor_minibatch(
                actor_unit=actor_unit,
                agent_id=agent_id,
                mb=mb,
                agent_states=agent_states,
                actions_by_agent=actions_by_agent,
                old_log_probs_by_agent=old_log_probs_by_agent,
                advantages_all=advantages_all,
            )

            actor_losses.append(actor_loss_i)
            actor_info_by_agent[agent_id] = actor_info_i

        actor_loss = torch.stack(actor_losses).mean()

        actor_unit.actor_net_optimiser.zero_grad()
        actor_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                list(actor_unit.actor_net.parameters()) + [actor_unit.log_std],
                self.max_grad_norm,
            )

        actor_unit.actor_net_optimiser.step()

        with torch.no_grad():
            actor_unit.log_std.clamp_(
                actor_unit.min_log_std,
                actor_unit.max_log_std,
            )

        post_update_kl_by_agent: dict[str, float] = {}

        for agent_id in active_agent_ids:
            kl_info = self._compute_post_update_kl(
                actor_unit=actor_unit,
                agent_id=agent_id,
                mb=mb,
                agent_states=agent_states,
                actions_by_agent=actions_by_agent,
                old_log_probs_by_agent=old_log_probs_by_agent,
            )

            actor_info_by_agent[agent_id].update(kl_info)
            post_update_kl_by_agent[agent_id] = kl_info["approx_kl"]

        group_early_stop = self.target_kl is not None and any(
            kl > self.target_kl for kl in post_update_kl_by_agent.values()
        )

        if group_early_stop:
            # For team_all, this stops the whole shared actor group.
            # For individual/team_critic, agent_ids has length 1, so this is
            # equivalent to per-agent KL early stopping.
            return (
                {agent_id: True for agent_id in agent_ids},
                actor_info_by_agent,
            )

        return (
            {agent_id: False for agent_id in agent_ids},
            actor_info_by_agent,
        )

    def _compute_values_and_last_values(
        self,
        global_states: torch.Tensor,
        next_global_states: torch.Tensor,
        dones_by_agent: dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        values_all = torch.zeros(
            (batch_size, self.num_agents),
            device=self.device,
        )

        last_values_all = torch.zeros(
            (self.num_agents,),
            device=self.device,
        )

        with torch.no_grad():
            for critic_id, critic_agent_ids in self.critic_id_to_agent_ids.items():
                critic_unit = self.learning_units[critic_id]

                critic_agent_indices = [
                    self.all_agent_ids.index(agent_id) for agent_id in critic_agent_ids
                ]

                values_i = critic_unit.critic_net(global_states)
                values_i = values_i.view(batch_size, len(critic_agent_ids))

                last_next_state = next_global_states[-1].unsqueeze(0)
                last_value_i = critic_unit.critic_net(last_next_state).view(-1)

                last_done_i = torch.stack(
                    [dones_by_agent[agent_id][-1] for agent_id in critic_agent_ids]
                ).bool()

                last_value_i = last_value_i * (~last_done_i).to(last_value_i.dtype)

                values_all[:, critic_agent_indices] = values_i
                last_values_all[critic_agent_indices] = last_value_i

        return values_all, last_values_all

    def _compute_gae_returns(
        self,
        rewards_by_agent: dict[str, torch.Tensor],
        dones_by_agent: dict[str, torch.Tensor],
        values_all: torch.Tensor,
        last_values_all: torch.Tensor,
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        advantages_all = torch.zeros(
            (batch_size, self.num_agents),
            device=self.device,
        )

        returns_all = torch.zeros(
            (batch_size, self.num_agents),
            device=self.device,
        )

        for agent_index, agent_id in enumerate(self.all_agent_ids):
            actor_id = self.agent_id_to_actor_id[agent_id]
            actor_unit = self.learning_units[actor_id]

            adv_i, ret_i = actor_unit._calculate_gae(
                rewards=rewards_by_agent[agent_id],
                dones=dones_by_agent[agent_id],
                values=values_all[:, agent_index],
                last_value=last_values_all[agent_index],
                gae_lambda=self.gae_lambda,
            )

            advantages_all[:, agent_index] = adv_i
            returns_all[:, agent_index] = ret_i

        adv_flat = advantages_all.view(-1)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std(unbiased=False) + 1e-8)

        advantages_all = adv_flat.view(batch_size, self.num_agents)

        return advantages_all, returns_all

    def _sample_training_batch(
        self,
        memory_buffer: MARLMemoryBuffer,
    ) -> MAPPOBatch | None:
        sample = memory_buffer.flush()
        batch_size = len(sample.experiences)

        if batch_size == 0:
            return None

        sample_tensor = memory_sampler.sample_to_tensors(
            sample,
            self.device,
        )

        old_log_probs_by_agent = {
            agent_id: torch.tensor(
                np.asarray(
                    [
                        experience.train_data["log_prob"][agent_id]
                        for experience in sample.experiences
                    ]
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for agent_id in self.all_agent_ids
        }

        rewards_by_agent = {
            agent_id: sample_tensor.reward[agent_id].squeeze(-1)
            for agent_id in self.all_agent_ids
        }

        dones_by_agent = {
            agent_id: sample_tensor.done[agent_id].squeeze(-1).float()
            for agent_id in self.all_agent_ids
        }

        return MAPPOBatch(
            batch_size=batch_size,
            global_states=sample_tensor.observation.global_state,
            next_global_states=sample_tensor.next_observation.global_state,
            agent_states=sample_tensor.observation.agent_states,
            actions_by_agent=sample_tensor.action,
            rewards_by_agent=rewards_by_agent,
            dones_by_agent=dones_by_agent,
            old_log_probs_by_agent=old_log_probs_by_agent,
        )

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        if self.parameter_sharing_scope not in {
            "individual",
            "team_critic",
            "team_all",
        }:
            raise ValueError(f"Invalid {self.parameter_sharing_scope=}")

        self.entropy_coef = self.entropy_scheduler.get_value(
            episode_context.training_step
        )

        samples = self._sample_training_batch(memory_buffer)

        if samples is None:
            return {}

        batch_size = samples.batch_size
        global_states = samples.global_states
        next_global_states = samples.next_global_states
        agent_states = samples.agent_states

        actions_by_agent = samples.actions_by_agent
        rewards_by_agent = samples.rewards_by_agent
        dones_by_agent = samples.dones_by_agent
        old_log_probs_by_agent = samples.old_log_probs_by_agent

        # ---------------------------------------------------------
        # Central critic values by critic ownership
        # ---------------------------------------------------------
        values_all, last_values_all = self._compute_values_and_last_values(
            global_states=global_states,
            next_global_states=next_global_states,
            dones_by_agent=dones_by_agent,
            batch_size=batch_size,
        )

        # ---------------------------------------------------------
        # GAE advantages and returns by agent
        # ---------------------------------------------------------
        advantages_all, returns_all = self._compute_gae_returns(
            rewards_by_agent=rewards_by_agent,
            dones_by_agent=dones_by_agent,
            values_all=values_all,
            last_values_all=last_values_all,
            batch_size=batch_size,
        )

        mb_size = min(self.minibatch_size, batch_size)

        agent_kl_early_stop = {agent_id: False for agent_id in self.all_agent_ids}

        agent_actor_sums: dict[str, dict[str, float]] = {
            agent_id: {} for agent_id in self.all_agent_ids
        }

        agent_max_kl = {agent_id: 0.0 for agent_id in self.all_agent_ids}

        agent_updates = {agent_id: 0 for agent_id in self.all_agent_ids}

        critic_sums: dict[str, dict[str, float]] = {
            critic_id: {} for critic_id in self.critic_id_to_agent_ids.keys()
        }

        critic_updates = {
            critic_id: 0 for critic_id in self.critic_id_to_agent_ids.keys()
        }

        # ---------------------------------------------------------
        # PPO epochs / minibatches
        # ---------------------------------------------------------
        for _ in range(self.updates_per_iteration):
            idx = torch.randperm(batch_size, device=self.device)

            for start in range(0, batch_size, mb_size):
                mb = idx[start : start + mb_size]

                # -------------------------------------------------
                # Actor updates remain per team agent.
                # -------------------------------------------------
                for actor_id, actor_agent_ids in self.actor_id_to_agent_ids.items():
                    actor_unit = self.learning_units[actor_id]

                    kl_updates, actor_info_by_agent = self._update_actor_minibatch(
                        actor_unit=actor_unit,
                        agent_ids=actor_agent_ids,
                        mb=mb,
                        agent_states=agent_states,
                        actions_by_agent=actions_by_agent,
                        old_log_probs_by_agent=old_log_probs_by_agent,
                        advantages_all=advantages_all,
                        agent_kl_early_stop=agent_kl_early_stop,
                    )

                    for agent_id, kl_stop in kl_updates.items():
                        agent_kl_early_stop[agent_id] = kl_stop

                    for agent_id, actor_info in actor_info_by_agent.items():
                        agent_updates[agent_id] += 1

                        for key, value in actor_info.items():
                            agent_actor_sums[agent_id][key] = agent_actor_sums[
                                agent_id
                            ].get(key, 0.0) + float(value)

                        if self.target_kl is not None:
                            agent_max_kl[agent_id] = max(
                                agent_max_kl[agent_id],
                                float(actor_info["approx_kl"]),
                            )

                # -------------------------------------------------
                # Critic updates are grouped by critic ownership.
                # -------------------------------------------------
                for critic_id, critic_agent_ids in self.critic_id_to_agent_ids.items():
                    critic_unit = self.learning_units[critic_id]

                    critic_info = self._update_critic_minibatch(
                        critic_unit=critic_unit,
                        critic_agent_ids=critic_agent_ids,
                        mb=mb,
                        global_states=global_states,
                        returns_all=returns_all,
                    )

                    critic_updates[critic_id] += 1

                    for key, value in critic_info.items():
                        critic_sums[critic_id][key] = critic_sums[critic_id].get(
                            key, 0.0
                        ) + float(value)

        # ---------------------------------------------------------
        # Logging
        # ---------------------------------------------------------
        info["entropy_coef"] = self.entropy_coef

        for critic_id, metric_sums in critic_sums.items():
            denom = max(critic_updates[critic_id], 1)

            for key, value in metric_sums.items():
                info[f"{critic_id}_{key}"] = value / denom

        with torch.no_grad():
            td_err = returns_all - values_all

            for i, agent_id in enumerate(self.all_agent_ids):
                info[f"{agent_id}_adv_mean"] = float(advantages_all[:, i].mean().item())
                info[f"{agent_id}_adv_std"] = float(
                    advantages_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_id}_ret_mean"] = float(returns_all[:, i].mean().item())
                info[f"{agent_id}_ret_std"] = float(
                    returns_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_id}_v_mean"] = float(values_all[:, i].mean().item())
                info[f"{agent_id}_v_std"] = float(
                    values_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_id}_td_mean"] = float(td_err[:, i].mean().item())
                info[f"{agent_id}_td_std"] = float(
                    td_err[:, i].std(unbiased=False).item()
                )
                info[f"{agent_id}_td_mae"] = float(td_err[:, i].abs().mean().item())

                y = returns_all[:, i]
                yhat = values_all[:, i]
                var_y = torch.var(y, unbiased=False)
                ev = 1.0 - torch.var(y - yhat, unbiased=False) / (var_y + 1e-8)
                info[f"{agent_id}_explained_var"] = float(ev.item())

                denom = max(agent_updates[agent_id], 1)

                info[f"{agent_id}_actor_updates"] = int(agent_updates[agent_id])
                info[f"{agent_id}_kl_early_stop"] = int(agent_kl_early_stop[agent_id])

                for key, value in agent_actor_sums[agent_id].items():
                    info[f"{agent_id}_{key}"] = value / denom

                if self.target_kl is not None:
                    info[f"{agent_id}_max_kl_seen"] = agent_max_kl[agent_id]

        # ---------------------------------------------------------
        # Aggregate actor metrics
        # ---------------------------------------------------------
        actor_metric_keys = {
            key
            for metric_sums in agent_actor_sums.values()
            for key in metric_sums.keys()
        }

        for key in actor_metric_keys:
            values = [
                info[f"{agent_id}_{key}"]
                for agent_id in self.all_agent_ids
                if f"{agent_id}_{key}" in info
            ]

            if values:
                info[f"mean_{key}"] = float(np.mean(values))

        # ---------------------------------------------------------
        # Aggregate critic metrics
        # ---------------------------------------------------------
        critic_metric_keys = {
            key for metric_sums in critic_sums.values() for key in metric_sums.keys()
        }

        for key in critic_metric_keys:
            values = [
                info[f"{critic_id}_{key}"]
                for critic_id in self.critic_id_to_agent_ids.keys()
                if f"{critic_id}_{key}" in info
            ]

            if values:
                info[f"mean_{key}"] = float(np.mean(values))
                info[f"std_{key}"] = float(np.std(values))
                info[f"max_{key}"] = float(np.max(values))
                info[f"min_{key}"] = float(np.min(values))

        # ---------------------------------------------------------
        # Aggregate value/return metrics
        # ---------------------------------------------------------
        for metric in [
            "ret_mean",
            "ret_std",
            "v_mean",
            "v_std",
            "td_mae",
            "explained_var",
        ]:
            values = [
                info[f"{agent_id}_{metric}"]
                for agent_id in self.all_agent_ids
                if f"{agent_id}_{metric}" in info
            ]

            if values:
                info[f"mean_{metric}"] = float(np.mean(values))
                info[f"std_{metric}"] = float(np.std(values))
                info[f"max_{metric}"] = float(np.max(values))
                info[f"min_{metric}"] = float(np.min(values))

        stopped = sum(int(x) for x in agent_kl_early_stop.values())
        info["num_agents_kl_stopped"] = stopped
        info["any_agent_kl_stopped"] = int(stopped > 0)
        info["all_agents_kl_stopped"] = int(stopped == self.num_agents)

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
