"""
MAPPO (Multi-Agent Proximal Policy Optimization) implementation notes
--------------------------------------------------------------------

Original Paper: https://arxiv.org/abs/2103.01955

Modernisation Notes
-------------------

This implementation aims to faithfully reproduce the original algorithmic
formulation as closely as possible while integrating with the CARES multi-agent
reinforcement learning framework.

The implementation preserves the core methodological components described in
the original work, including centralized training with decentralized execution (CTDE),
shared replay/rollout sampling where appropriate, and the original actor-critic
optimization structure.

To support broader experimentation and consistent evaluation across MARL algorithms,
the implementation additionally exposes configurable parameter-sharing and
critic-sharing modes through a unified framework abstraction. These extensions are
designed to remain compatible with the canonical default configuration used
in the original algorithm formulation.

This implementation preserves the original MAPPO centralized-training /
decentralized-execution (CTDE) formulation while adopting several modern
training conventions commonly used in contemporary MARL frameworks.

MAPPO implementation overview
-----------------------------

MAPPO extends PPO to the multi-agent setting using:

    - centralized value critics,
    - decentralized stochastic actors,
    - generalized advantage estimation (GAE),
    - clipped PPO policy optimisation,
    - value target normalization,
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

Value Normalization
    This implementation uses running return normalization for critic targets.

    Critics predict normalized scalar values while:

    - GAE,
    - logging,
    - and rollout statistics

    operate in the original return scale.

    The MAPPO paper reports value normalization as an important stabilizing
    component, especially in cooperative MPE environments.

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

MAPPO uses scalar centralized value critics.

Each critic learning unit estimates one scalar value:

        V(s) -> scalar return estimate

The critic receives the global environment state during training and is
optimized using generalized advantage estimation (GAE) returns.

For each critic learning unit:

    1. Compute centralized scalar value estimates from the global state.
    2. Compute per-agent returns and advantages using GAE.
    3. Aggregate the controlled agents' returns into a scalar critic target.
    4. Regress the critic against that scalar target.

individual:
    each critic belongs to one environment agent and therefore optimizes only
    that agent's returns.

team_critic:
    one centralized scalar critic is shared across the team while actors remain
    independent per environment agent.

    The critic target is formed from the controlled agents' aggregated returns.

team_all:
    one centralized scalar critic is shared across the entire actor-sharing
    team.

    The critic therefore learns one shared cooperative team value estimate.

Shared scalar critics are the canonical MAPPO formulation used in the original
paper for cooperative environments with shared team rewards.

This implementation additionally supports more general reward structures by
aggregating the controlled agents' returns into a single scalar critic target
for each critic learning unit.

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

Advantage Normalization
-----------------------

Advantages are normalized according to the actor learning unit update group.

individual:
    Advantages are normalized per environment agent.

team_critic:
    Advantages are normalized per environment agent.

team_all:
    Advantages are normalized jointly across all agents controlled by the
    shared actor.

For homogeneous cooperative environments with one shared actor, this becomes
equivalent to the global advantage normalization used in the original MAPPO
implementation.

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
        a common configuration for cooperative multi-agent tasks:
            - critics learn a centralized cooperative objective,
            - while actors remain agent-specific and relatively stable.

        This typically provides a good balance between:
            - coordination,
            - stability,
            - and optimization simplicity.

    team_all:
        closest to original MAPPO

        the default configuration for MAPPO in the original paper:
            - both actor and critic are shared across the team,
            - and PPO updates optimise a coupled team objective.

        This configuration is often most effective for homogeneous agents.

    individual:
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
from cares_reinforcement_learning.algorithm.value_normaliser import ValueNormaliser
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

        self.huber_delta = config.huber_delta

        self.value_normalisers = {
            critic_id: ValueNormaliser(
                shape=(),
                device=self.device,
            )
            for critic_id in self.critic_id_to_agent_ids
        }

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
        mb: torch.Tensor,
        global_states: torch.Tensor,
        normalised_returns_i: torch.Tensor,
        old_values_i: torch.Tensor,
    ) -> dict[str, Any]:
        v_pred = critic_unit.critic_net(global_states[mb]).view(-1)

        v_targ = normalised_returns_i[mb]
        old_v_pred = old_values_i[mb]

        v_pred_clipped = old_v_pred + torch.clamp(
            v_pred - old_v_pred,
            -critic_unit.eps_clip,
            critic_unit.eps_clip,
        )

        value_loss_unclipped = F.huber_loss(
            v_pred, v_targ, reduction="none", delta=self.huber_delta
        )

        value_loss_clipped = F.huber_loss(
            v_pred_clipped, v_targ, reduction="none", delta=self.huber_delta
        )

        critic_loss = torch.max(
            value_loss_unclipped,
            value_loss_clipped,
        ).mean()

        critic_unit.critic_net_optimiser.zero_grad()
        critic_loss.backward()

        if self.max_grad_norm is not None:
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
                "value_clip_frac": ((v_pred - old_v_pred).abs() > critic_unit.eps_clip)
                .float()
                .mean()
                .item(),
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

        logp = logp.view(-1)
        old_logp_mb = old_logp_mb.view(-1)
        advantages_mb = advantages_mb.view(-1)

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
            new_logp = new_logp.view(-1)
            old_logp_mb = old_logp_mb.view(-1)

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
        actor_kl_early_stop: bool,
    ) -> tuple[bool, dict[str, dict[str, Any]]]:
        actor_losses: list[torch.Tensor] = []
        actor_info_by_agent: dict[str, dict[str, Any]] = {}

        if actor_kl_early_stop:
            return True, actor_info_by_agent

        active_agent_ids = agent_ids

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

        return group_early_stop, actor_info_by_agent

    def _normalise_returns_by_critic(
        self,
        returns_all: torch.Tensor,
    ) -> torch.Tensor:
        normalised_returns_all = torch.zeros_like(returns_all)

        for critic_id, critic_agent_ids in self.critic_id_to_agent_ids.items():
            critic_agent_indices = [
                self.all_agent_ids.index(agent_id) for agent_id in critic_agent_ids
            ]

            # Scalar team/critic target.
            critic_returns = returns_all[:, critic_agent_indices].mean(dim=1)

            value_normaliser = self.value_normalisers[critic_id]
            value_normaliser.update(critic_returns)

            normalised_critic_returns = value_normaliser.normalise(critic_returns)

            # Broadcast the scalar critic target back onto the agents owned by
            # this critic so the existing minibatch update indexing stays simple.
            normalised_returns_all[:, critic_agent_indices] = (
                normalised_critic_returns.unsqueeze(-1)
            )

        return normalised_returns_all

    def _compute_values_and_last_values(
        self,
        global_states: torch.Tensor,
        next_global_states: torch.Tensor,
        dones_by_agent: dict[str, torch.Tensor],
        batch_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        values_all = torch.zeros(
            (batch_size, self.num_agents),
            device=self.device,
        )

        last_values_all = torch.zeros(
            (self.num_agents,),
            device=self.device,
        )

        old_normalised_values_by_critic: dict[str, torch.Tensor] = {}

        with torch.no_grad():
            for critic_id, critic_agent_ids in self.critic_id_to_agent_ids.items():
                critic_unit = self.learning_units[critic_id]

                critic_agent_indices = [
                    self.all_agent_ids.index(agent_id) for agent_id in critic_agent_ids
                ]

                # -----------------------------------------------------
                # Current-state values
                # -----------------------------------------------------
                # The critic predicts normalised scalar values.
                # Store the normalised predictions for PPO value clipping,
                # and denormalise copies for GAE / return computation.
                value_normaliser = self.value_normalisers[critic_id]

                values_i_norm = (
                    critic_unit.critic_net(global_states).view(batch_size).detach()
                )
                values_i = value_normaliser.denormalise(values_i_norm)

                # save frozen normalised values for value clipping
                old_normalised_values_by_critic[critic_id] = values_i_norm

                last_next_state = next_global_states[-1].unsqueeze(0)
                last_value_i_norm = (
                    critic_unit.critic_net(last_next_state).view(()).detach()
                )
                last_value_i = value_normaliser.denormalise(last_value_i_norm)

                last_done_i = (
                    torch.stack(
                        [dones_by_agent[agent_id][-1] for agent_id in critic_agent_ids]
                    )
                    .amax()
                    .bool()
                )

                last_value_i = last_value_i * (~last_done_i).to(last_value_i.dtype)

                values_all[:, critic_agent_indices] = values_i.unsqueeze(-1)
                last_values_all[critic_agent_indices] = last_value_i

        return values_all, last_values_all, old_normalised_values_by_critic

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

        # ---------------------------------------------------------
        # Compute raw GAE advantages and returns per environment agent
        # ---------------------------------------------------------
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

        # ---------------------------------------------------------
        # Normalize advantages by actor learning unit
        # ---------------------------------------------------------
        #
        # The normalization group should match the PPO actor update group:
        #
        # individual:
        #   one actor per agent -> normalize each agent separately
        #
        # team_critic:
        #   one actor per agent, shared critic -> normalize each agent separately
        #
        # team_all:
        #   one shared actor per team -> normalize over all agents controlled
        #   by that shared actor/team
        #
        # For official shared-policy MAPPO in a single cooperative team, this is
        # equivalent to global normalization across all agents.
        for actor_id, actor_agent_ids in self.actor_id_to_agent_ids.items():
            actor_agent_indices = [
                self.all_agent_ids.index(agent_id) for agent_id in actor_agent_ids
            ]

            adv_i = advantages_all[:, actor_agent_indices]

            advantages_all[:, actor_agent_indices] = (adv_i - adv_i.mean()) / (
                adv_i.std(unbiased=False) + 1e-8
            )

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

        rewards_by_agent = {
            agent_id: sample_tensor.reward[agent_id].squeeze(-1)
            for agent_id in self.all_agent_ids
        }

        dones_by_agent = {
            agent_id: sample_tensor.done[agent_id].squeeze(-1).float()
            for agent_id in self.all_agent_ids
        }

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
        values_all, last_values_all, old_values_by_critic = (
            self._compute_values_and_last_values(
                global_states=global_states,
                next_global_states=next_global_states,
                dones_by_agent=dones_by_agent,
                batch_size=batch_size,
            )
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
        normalised_returns_all = self._normalise_returns_by_critic(returns_all)

        mb_size = min(self.minibatch_size, batch_size)

        actor_kl_early_stop = {
            actor_id: False for actor_id in self.actor_id_to_agent_ids
        }

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

                    actor_kl_stop, actor_info_by_agent = self._update_actor_minibatch(
                        actor_unit=actor_unit,
                        agent_ids=actor_agent_ids,
                        mb=mb,
                        agent_states=agent_states,
                        actions_by_agent=actions_by_agent,
                        old_log_probs_by_agent=old_log_probs_by_agent,
                        advantages_all=advantages_all,
                        actor_kl_early_stop=actor_kl_early_stop[actor_id],
                    )

                    actor_kl_early_stop[actor_id] = (
                        actor_kl_early_stop[actor_id] or actor_kl_stop
                    )

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

                    critic_agent_indices = [
                        self.all_agent_ids.index(agent_id)
                        for agent_id in critic_agent_ids
                    ]

                    normalised_returns_i = normalised_returns_all[
                        :, critic_agent_indices
                    ].mean(dim=1)

                    old_values_i = old_values_by_critic[critic_id]

                    critic_info = self._update_critic_minibatch(
                        critic_unit=critic_unit,
                        mb=mb,
                        global_states=global_states,
                        normalised_returns_i=normalised_returns_i,
                        old_values_i=old_values_i,
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
                info[f"{critic_id}.{key}"] = value / denom

        with torch.no_grad():
            td_err = returns_all - values_all

            for i, agent_id in enumerate(self.all_agent_ids):
                info[f"{agent_id}.adv_mean"] = float(advantages_all[:, i].mean().item())
                info[f"{agent_id}.adv_std"] = float(
                    advantages_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_id}.ret_mean"] = float(returns_all[:, i].mean().item())
                info[f"{agent_id}.ret_std"] = float(
                    returns_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_id}.v_mean"] = float(values_all[:, i].mean().item())
                info[f"{agent_id}.v_std"] = float(
                    values_all[:, i].std(unbiased=False).item()
                )

                info[f"{agent_id}.td_mean"] = float(td_err[:, i].mean().item())
                info[f"{agent_id}.td_std"] = float(
                    td_err[:, i].std(unbiased=False).item()
                )
                info[f"{agent_id}.td_mae"] = float(td_err[:, i].abs().mean().item())

                y = returns_all[:, i]
                yhat = values_all[:, i]
                var_y = torch.var(y, unbiased=False)
                ev = 1.0 - torch.var(y - yhat, unbiased=False) / (var_y + 1e-8)
                info[f"{agent_id}.explained_var"] = float(ev.item())

                denom = max(agent_updates[agent_id], 1)

                info[f"{agent_id}.actor_updates"] = int(agent_updates[agent_id])
                info[f"{agent_id}.kl_early_stop"] = int(
                    actor_kl_early_stop[self.agent_id_to_actor_id[agent_id]]
                )

                for key, value in agent_actor_sums[agent_id].items():
                    info[f"{agent_id}.{key}"] = value / denom

                if self.target_kl is not None:
                    info[f"{agent_id}.max_kl_seen"] = agent_max_kl[agent_id]

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
                info[f"{agent_id}.{key}"]
                for agent_id in self.all_agent_ids
                if f"{agent_id}.{key}" in info
            ]

            if values:
                info[f"mean.{key}"] = float(np.mean(values))
                info[f"std.{key}"] = float(np.std(values))
                info[f"max.{key}"] = float(np.max(values))
                info[f"min.{key}"] = float(np.min(values))

        # ---------------------------------------------------------
        # Aggregate critic metrics
        # ---------------------------------------------------------
        critic_metric_keys = {
            key for metric_sums in critic_sums.values() for key in metric_sums.keys()
        }

        for key in critic_metric_keys:
            values = [
                info[f"{critic_id}.{key}"]
                for critic_id in self.critic_id_to_agent_ids.keys()
                if f"{critic_id}.{key}" in info
            ]

            if values:
                info[f"mean.{key}"] = float(np.mean(values))
                info[f"std.{key}"] = float(np.std(values))
                info[f"max.{key}"] = float(np.max(values))
                info[f"min.{key}"] = float(np.min(values))

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
                info[f"{agent_id}.{metric}"]
                for agent_id in self.all_agent_ids
                if f"{agent_id}.{metric}" in info
            ]

            if values:
                info[f"mean.{metric}"] = float(np.mean(values))
                info[f"std.{metric}"] = float(np.std(values))
                info[f"max.{metric}"] = float(np.max(values))
                info[f"min.{metric}"] = float(np.min(values))

        stopped_actor_units = sum(int(x) for x in actor_kl_early_stop.values())

        stopped_agents = sum(
            int(actor_kl_early_stop[self.agent_id_to_actor_id[agent_id]])
            for agent_id in self.all_agent_ids
        )

        info["num_actor_units_kl_stopped"] = stopped_actor_units
        info["any_actor_unit_kl_stopped"] = int(stopped_actor_units > 0)
        info["all_actor_units_kl_stopped"] = int(
            stopped_actor_units == len(actor_kl_early_stop)
        )

        info["num_agents_kl_stopped"] = stopped_agents
        info["any_agent_kl_stopped"] = int(stopped_agents > 0)
        info["all_agents_kl_stopped"] = int(stopped_agents == self.num_agents)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for learning_unit_id, learning_unit in self.learning_units.items():
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_{learning_unit_id}_checkpoint"
            learning_unit.save_models(agent_filepath, agent_filename)

        torch.save(
            {
                critic_id: {
                    "mean": value_normaliser.mean.detach().cpu(),
                    "var": value_normaliser.var.detach().cpu(),
                    "count": value_normaliser.count.detach().cpu(),
                }
                for critic_id, value_normaliser in self.value_normalisers.items()
            },
            os.path.join(filepath, "value_normalisers.pht"),
        )

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for learning_unit_id, learning_unit in self.learning_units.items():
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_{learning_unit_id}_checkpoint"
            learning_unit.load_models(agent_filepath, agent_filename)

        normaliser_path = os.path.join(filepath, "value_normalisers.pht")
        normaliser_state = torch.load(normaliser_path, map_location=self.device)

        for critic_id, state in normaliser_state.items():
            if critic_id not in self.value_normalisers:
                continue

            self.value_normalisers[critic_id].mean = state["mean"].to(self.device)
            self.value_normalisers[critic_id].var = state["var"].to(self.device)
            self.value_normalisers[critic_id].count = state["count"].to(self.device)

        logging.info("models and optimisers have been loaded...")
