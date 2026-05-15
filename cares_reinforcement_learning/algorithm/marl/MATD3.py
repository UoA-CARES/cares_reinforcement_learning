"""
MATD3 implementation overview
-----------------------------

MATD3 extends MADDPG with the core TD3 stabilisation mechanisms:

    - twin critics,
    - delayed actor updates,
    - and target policy smoothing.

This implementation additionally supports several multi-agent coordination and
parameter-sharing variants through two main configuration options:

    parameter_sharing_scope:
        "individual" -> one DDPG learning unit per environment agent (default)
        "team_critic" -> one shared critic per team, separate actor per agent
        "team_all"   -> one shared DDPG learning unit per team of agents
                        when using team sharing, update the shared actor by
                        replacing all controlled agents' replay actions with
                        current actor outputs at the same time (coupled)

Terminology
-----------

Environment agent:
    An agent that exists in the environment, e.g.
        adversary_0, adversary_1, agent_0

Learning unit:
    A trainable TD3 bundle containing:
        actor, twin critics, target networks, optimisers, and exploration noise.

    In individual mode:
        each environment agent has its own learning unit.

    In team mode:
        all agents in the same team share one learning unit.

Controlled agents:
    The environment agents assigned to a learning unit.

Centralised training, decentralised execution
---------------------------------------------

MATD3 follows the CTDE (centralised training, decentralised execution)
paradigm inherited from MADDPG.

Training:
    critics receive:
        - the global state,
        - and the joint action of all environment agents.

Execution:
    actors receive only the local observation of the environment agent they are
    producing an action for.

Even in team mode, actions are still produced independently per environment
agent. The shared team actor is simply reused across multiple agents belonging
to the same team.

TD3 extensions
--------------

Twin critics:
    MATD3 uses two critics per learning unit and minimises the target over both
    critics:

        target_q = min(Q1_target, Q2_target)

    This reduces overestimation bias compared to MADDPG.

Delayed actor updates:
    Actor and target-network updates are performed less frequently than critic
    updates using:

        policy_update_freq

    This allows critics to stabilise before the actor is updated.

Target policy smoothing:
    Gaussian noise is added to NEXT target actions during critic target
    computation:

        a_next = pi_target(o_next) + noise

    The noise is clipped and only applied to target actions used in the TD
    target computation.

    This discourages critics from exploiting narrow peaks in the learned value
    function and improves stability.

Shared replay sampling
----------------------

A single replay minibatch is sampled once per training step and reused for all
learning-unit updates.

This keeps all critics and actors training against the same set of joint
transitions during that iteration. It also reduces sampling overhead and makes
diagnostics easier to compare across learning units.

Critic update
-------------

For each learning unit:

    1. Build target actions for every environment agent using target actors.
    2. Apply TD3 target policy smoothing noise to NEXT actions.
    3. Flatten the noisy joint target action into shape (B, N * action_dim).
    4. Compute the TD target using the minimum target critic:

            y = r_i + gamma * (1 - done_i) * min(Q1_target, Q2_target)

    5. Regress both critics against the replay-buffer joint action:

            Q1_i(s, a_replay_joint)
            Q2_i(s, a_replay_joint)

In individual mode, r_i and done_i are the controlled agent's own reward/done.

In team mode, r_i and done_i are aggregated across the controlled agents.

Actor update
------------

The actor update follows deterministic policy gradient logic inherited from
MADDPG.

Replay actions are used as a fixed baseline joint action. The current actor's
output replaces one or more controlled agents' replay actions depending on the
configured update mode. The critic then evaluates the resulting counterfactual
joint action.

This answers the question:

    "If this learning unit changed the actions of the agents it controls, while
    the rest of the sampled transition stayed fixed, would the critic assign a
    higher value?"

Only the first critic is used for actor optimisation, matching standard TD3
practice.

No joint action resampling
--------------------------

TD3 stochasticity is confined to target policy smoothing applied to NEXT target
actions.

Current replay actions from other agents are intentionally kept fixed during
actor optimisation. This reduces variance and preserves the counterfactual-style
deterministic policy gradient structure inherited from MADDPG.

Practical notes
---------------

For cooperative environments such as simple_spread:

    team_critic:
        often a strong default because:
            - the shared team critic learns a cooperative objective,
            - while actors remain agent-specific and relatively conservative.

        This typically provides a good balance between:
            - coordination,
            - stability,
            - and optimisation simplicity.

    team_all:
        can produce stronger coordinated behaviour because:
            - both the actor and critic are shared across the team,
            - and actor updates optimise a fully coupled team objective.

        However, this configuration is usually more sensitive to:
            - actor learning rate,
            - replay variance,
            - and critic instability.

        Lower actor learning rates and larger batch sizes are often beneficial.

    individual:
        closest to the original MATD3 formulation.

        Each agent learns:
            - its own actor,
            - and its own critic.

        This provides the strongest agent specialisation, but critics optimise
        more local objectives and may learn cooperative coordination more slowly
        in fully cooperative environments.

Compared to MADDPG, MATD3 is typically:
    - more stable,
    - less sensitive to critic overestimation,
    - and better behaved under larger replay buffers or noisier environments.
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
        env_teams: dict[str, list[str]],
        agent_id_to_actor_id: dict[str, str],
        actor_id_to_agent_ids: dict[str, list[str]],
        agent_id_to_critic_id: dict[str, str],
        critic_id_to_agent_ids: dict[str, list[str]],
        config: MATD3Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        # Physical trainable containers.
        #
        # Each learning unit is a DDPG bundle containing:
        #   - actor
        #   - target actor
        #   - critic
        #   - target critic
        #   - optimisers
        #   - exploration schedule
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
            learning_unit_id = self.agent_id_to_actor_id[agent_id]
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
        actor_unit: TD3,
        critic_unit: TD3,
        controlled_agent_ids: list[str],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        replay_actions: torch.Tensor,
    ) -> tuple[list[torch.Tensor], torch.Tensor, torch.Tensor]:
        # Build a counterfactual joint action for actor optimisation.
        #
        # Replay actions act as the fixed baseline joint action.
        #
        # Controlled agents' replay actions are replaced with current
        # policy outputs from the actor under optimisation.
        #
        # Example:
        #
        #   replay:
        #       [a_0_replay, a_1_replay, a_2_replay]
        #
        #   uncoupled:
        #       [pi(o_0), a_1_replay, a_2_replay]
        #
        #   coupled:
        #       [pi(o_0), pi(o_1), pi(o_2)]
        #
        # The critic then evaluates the resulting counterfactual joint action.
        batch_size = global_states.shape[0]
        actions_all = replay_actions.clone()

        actions_i_all = []

        for controlled_agent_id in controlled_agent_ids:
            agent_index = self.all_agent_ids.index(controlled_agent_id)

            obs_i = obs_tensors[controlled_agent_id]
            actions_i = actor_unit.actor_net(obs_i)

            actions_all[:, agent_index, :] = actions_i
            actions_i_all.append(actions_i)

        joint_actions_flat = actions_all.reshape(batch_size, -1)

        actor_q_values, _ = critic_unit.critic_net(
            global_states,
            joint_actions_flat,
        )

        actor_objective = -actor_q_values.mean()

        return actions_i_all, actor_q_values, actor_objective

    def _update_actor(
        self,
        actor_unit: TD3,
        critic_unit: TD3,
        controlled_agent_ids: list[str],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        replay_actions: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        # Coupled updates are only meaningful when multiple agents share
        # the SAME actor parameters.
        #
        # team_all:
        #     shared actor per team
        #     -> evaluate all controlled agents' policy actions together
        #
        # individual / team_critic:
        #     separate actor per agent
        #     -> use counterfactual one-agent-at-a-time updates
        #
        # This mirrors the original MADDPG deterministic policy gradient logic.
        #
        # Uncoupled actor update:
        #     evaluate one controlled agent's new action at a time
        #     average losses
        #
        #     Q_1 = Q_team(s, [π(o_1), replay(a_2), replay(a_3)])
        #     Q_2 = Q_team(s, [replay(a_1), π(o_2), replay(a_3)])
        #     Q_3 = Q_team(s, [replay(a_1), replay(a_2), π(o_3)])
        #
        #     actor_loss = mean(-Q_1.mean(), -Q_2.mean(), -Q_3.mean())
        #
        # Coupled actor update:
        #     evaluate all controlled agents' new actions together
        #     one team loss
        #
        #     actor_loss = -Q_team(s, [π(o_1), π(o_2), π(o_3)]).mean()
        use_coupled_update = self.parameter_sharing_scope == "team_all"

        if use_coupled_update:
            contribution_groups = [controlled_agent_ids]
        else:
            contribution_groups = [[agent_id] for agent_id in controlled_agent_ids]

        actions_all: list[torch.Tensor] = []
        contribution_actions_all: list[list[torch.Tensor]] = []
        actor_q_values_all: list[torch.Tensor] = []
        actor_objectives_all: list[torch.Tensor] = []

        for contribution_agent_ids in contribution_groups:
            actions_i_all, actor_q_values, actor_objective = (
                self._build_actor_contribution(
                    actor_unit=actor_unit,
                    critic_unit=critic_unit,
                    controlled_agent_ids=contribution_agent_ids,
                    obs_tensors=obs_tensors,
                    global_states=global_states,
                    replay_actions=replay_actions,
                )
            )

            actions_all.extend(actions_i_all)
            contribution_actions_all.append(actions_i_all)
            actor_q_values_all.append(actor_q_values)
            actor_objectives_all.append(actor_objective)

        actor_loss = torch.stack(actor_objectives_all).mean()

        actions_cat = torch.cat(actions_all, dim=0)
        actor_q_cat = torch.cat(actor_q_values_all, dim=0)

        dq_da_values: list[torch.Tensor] = []

        for actor_q_values_i, contribution_actions in zip(
            actor_q_values_all,
            contribution_actions_all,
        ):
            for actions_i in contribution_actions:
                dq_da = torch.autograd.grad(
                    outputs=-actor_q_values_i.mean(),
                    inputs=actions_i,
                    retain_graph=True,
                    create_graph=False,
                )[0]
                dq_da_values.append(dq_da.detach())

        dq_da_cat = torch.cat(dq_da_values, dim=0)

        actor_unit.actor_net_optimiser.zero_grad()
        actor_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                actor_unit.actor_net.parameters(),
                self.max_grad_norm,
            )

        actor_unit.actor_net_optimiser.step()

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

        # Build NEXT actions using the actor assigned to each env agent.
        #
        # individual:
        #     each agent uses its own target actor
        #
        # team_critic:
        #     each agent still uses its own target actor
        #
        # team_all:
        #     multiple env agents may reuse the same shared target actor
        next_actions: dict[str, torch.Tensor] = {}

        for agent_id in self.all_agent_ids:
            actor_id = self.agent_id_to_actor_id[agent_id]
            actor_unit = self.learning_units[actor_id]

            obs_next_i = sample_tensor.next_observation.agent_states[agent_id]

            with torch.no_grad():
                with fnc.evaluating(actor_unit.target_actor_net):
                    next_actions[agent_id] = actor_unit.target_actor_net(obs_next_i)

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

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:
        self.learn_counter += 1
        info: dict[str, Any] = {}

        # ---------------------------------------------------------
        # Update action noise schedules
        #
        # Noise is actor-owned, so only actor units need meaningful
        # action-noise values.
        # ---------------------------------------------------------
        for actor_id in self.actor_id_to_agent_ids.keys():
            actor_unit = self.learning_units[actor_id]

            actor_unit.action_noise = actor_unit.action_noise_scheduler.get_value(
                episode_context.training_step
            )

            info[f"action_noise_{actor_id}"] = float(actor_unit.action_noise)

        # Update TD3 target policy smoothing noise (decayed over training)
        self.policy_noise = self.policy_noise_scheduler.get_value(
            episode_context.training_step
        )
        info["current_policy_noise"] = self.policy_noise

        # ---------------------------------------------------------
        # Sample ONCE for all learning units
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across updates.
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

        update_actor = self.learn_counter % self.policy_update_freq == 0
        actor_info: dict[str, Any] = {}

        # ---------------------------------------------------------
        # Critic updates are grouped by CRITIC ownership, not actor ownership.
        #
        # individual:
        #     one critic update per agent
        #
        # team_critic:
        #     one critic update per team
        #
        # team_all:
        #     one critic update per team
        # ---------------------------------------------------------
        for critic_id, critic_agent_ids in self.critic_id_to_agent_ids.items():
            critic_unit = self.learning_units[critic_id]

            # ---------------------------------------------------------
            # Build learning-unit rewards/dones from controlled agents
            # individual: direct per-agent tensors
            # team: aggregate across controlled agents
            # ---------------------------------------------------------
            if self.parameter_sharing_scope == "individual":
                controlled_agent_id = critic_agent_ids[0]
                learning_unit_rewards = rewards_by_agent[controlled_agent_id]
                learning_unit_dones = dones_by_agent[controlled_agent_id]
            else:
                # Team critics optimise a shared cooperative objective.
                #
                # Rewards:
                #     mean reward across the controlled agents
                #
                # Dones:
                #     max(done_i)
                #
                # This means the team transition is considered terminal if ANY
                # controlled agent terminates.
                #
                # For fully cooperative environments this produces a team-level
                # learning signal while preserving centralized critic structure.
                learning_unit_rewards = torch.stack(
                    [rewards_by_agent[a] for a in critic_agent_ids],
                    dim=1,
                ).mean(dim=1)

                learning_unit_dones = torch.stack(
                    [dones_by_agent[a] for a in critic_agent_ids],
                    dim=1,
                ).amax(dim=1)

            with torch.no_grad():
                info[f"{critic_id}_reward_mean"] = learning_unit_rewards.mean().item()
                info[f"{critic_id}_done_frac"] = (
                    learning_unit_dones.float().mean().item()
                )

            critic_info = self._update_critic(
                learning_unit=critic_unit,
                global_states=global_states,
                joint_actions=joint_actions,
                rewards_i=learning_unit_rewards,
                next_global_states=next_global_states,
                next_actions_noisy=next_actions_noisy,
                dones_i=learning_unit_dones,
            )

            info.update({f"{critic_id}_{k}": v for k, v in critic_info.items()})

        # ---------------------------------------------------------
        # Actor update
        #
        # individual sharing:
        #     one actor update per environment agent.
        #
        # team + individual actor update:
        #     evaluate one controlled agent contribution at a time.
        #
        # team + joint actor update:
        #     evaluate all controlled agents together using one
        #     shared team objective.
        # ---------------------------------------------------------
        if update_actor:
            for actor_id, actor_agent_ids in self.actor_id_to_agent_ids.items():
                actor_unit = self.learning_units[actor_id]

                critic_id = self.agent_id_to_critic_id[actor_agent_ids[0]]
                critic_unit = self.learning_units[critic_id]

                actor_info = self._update_actor(
                    actor_unit=actor_unit,
                    critic_unit=critic_unit,
                    controlled_agent_ids=actor_agent_ids,
                    obs_tensors=agent_states,
                    global_states=global_states,
                    replay_actions=actions,
                )

                info.update({f"{actor_id}_{k}": v for k, v in actor_info.items()})

        # ---------------------------------------------------------
        # Target network updates (same cadence as actor updates)
        # ---------------------------------------------------------
        if update_actor:
            for learning_unit in self.learning_units.values():
                learning_unit.update_target_networks()

        # ---------------------------------------------------------
        # Cross-learning-unit diagnostics
        # ---------------------------------------------------------
        metrics = list(critic_info.keys())
        if update_actor:
            metrics += list(actor_info.keys())

        for metric in metrics:
            values = [
                value for key, value in info.items() if key.endswith(f"_{metric}")
            ]
            info[f"mean_{metric}"] = float(np.mean(values))
            info[f"std_{metric}"] = float(np.std(values))
            info[f"max_{metric}"] = float(np.max(values))
            info[f"min_{metric}"] = float(np.min(values))

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
