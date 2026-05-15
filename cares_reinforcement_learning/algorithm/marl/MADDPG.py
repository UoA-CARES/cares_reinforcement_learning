"""
MADDPG (Multi-Agent DDPG) implementation notes
---------------------------------------------

Original Paper: https://arxiv.org/pdf/1706.02275

Modernisation Notes
-------------------

This implementation preserves the original MADDPG centralized-training /
decentralized-execution (CTDE) formulation while adopting several modern
training conventions commonly used in contemporary MARL frameworks.

Implementation overview
----------

This implementation supports several MADDPG-style training variants through two
main configuration options:

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
    A trainable DDPG bundle containing:
        actor, critic, target_actor, target_critic, optimisers, noise schedule

    In individual mode:
        each environment agent has its own learning unit.

    In team mode:
        all agents in the same team share one learning unit.

Controlled agents:
    The environment agents assigned to a learning unit.

Centralised training, decentralised execution
---------------------------------------------

MADDPG follows CTDE:

    Training:
        critics receive the global state and the joint action of all agents.

    Execution:
        each actor receives only the local observation of the environment agent
        it is producing an action for.

Even in team mode, actions are still produced per environment agent. The shared
team actor is simply reused across multiple agents in the same team.

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
    2. Flatten the target joint action into shape (B, N * action_dim).
    3. Compute the TD target:

            y = r_i + gamma * (1 - done_i) * Q_target(s_next, a_next_joint)

    4. Regress the critic against the replay-buffer joint action:

            Q_i(s, a_replay_joint)

In individual mode, r_i and done_i are the controlled agent's own reward/done.

In team mode, r_i and done_i are aggregated over the controlled agents.

Actor update
------------

The actor update follows the deterministic policy gradient logic used by MADDPG.

Replay actions are used as a fixed baseline joint action. The current actor's
output replaces one or more controlled agents' replay actions depending on the
configured update mode. The critic then evaluates the resulting counterfactual
joint action.

This answers the question:

    "If this learning unit changed the actions of the agents it controls, while
    the rest of the sampled transition stayed fixed, would the critic assign a
    higher value?"

M3DDPG extension
----------------

If enabled, M3DDPG-style adversarial action perturbations are applied to the
other agents' actions during critic/actor updates.

The controlled agents' own actions are left unperturbed. This trains the critic
and actor under a more conservative assumption about the behaviour of other
agents.

ERNIE extension
---------------

If enabled, ERNIE-style observation perturbation regularisation is applied to
the actor.

The actor is encouraged to produce similar actions for small adversarial
perturbations of its observation. This can improve policy smoothness and
robustness.

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
        closest to the original MADDPG formulation.

        Each agent learns:
            - its own actor,
            - and its own critic.

        This provides the strongest agent specialisation, but critics optimise
        more local objectives and may learn cooperative coordination more slowly
        in fully cooperative environments.
"""

import logging
import os
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import MADDPGConfig
from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


@dataclass(frozen=True, slots=True)
class MADDPGBatch:
    global_states: torch.Tensor
    next_global_states: torch.Tensor
    agent_states: dict[str, torch.Tensor]
    next_agent_states: dict[str, torch.Tensor]

    actions_by_agent: dict[str, torch.Tensor]
    rewards_by_agent: dict[str, torch.Tensor]
    dones_by_agent: dict[str, torch.Tensor]

    actions: torch.Tensor  # (B, N, A)
    rewards: torch.Tensor  # (B, N, 1)
    dones: torch.Tensor  # (B, N, 1)
    next_actions: torch.Tensor  # (B, N, A)
    joint_actions: torch.Tensor  # (B, N * A)


class MADDPG(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        learning_units: dict[str, DDPG],
        all_agent_ids: list[str],
        env_teams: dict[str, list[str]],
        agent_id_to_actor_id: dict[str, str],
        actor_id_to_agent_ids: dict[str, list[str]],
        agent_id_to_critic_id: dict[str, str],
        critic_id_to_agent_ids: dict[str, list[str]],
        config: MADDPGConfig,
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

        self.max_grad_norm = config.max_grad_norm

        # M3DDPG adversarial perturbation scale
        self.use_m3 = config.use_m3
        self.m3_alpha = config.m3_alpha

        # ERNIE adversarial regularization
        self.use_ernie = config.use_ernie
        self.ernie_lambda = config.ernie_lambda
        self.ernie_eps = config.ernie_eps
        self.ernie_k_steps = config.ernie_k_steps
        self.ernie_norm = config.ernie_norm

        self.ernie_step_size = (
            self.ernie_eps / self.ernie_k_steps if self.ernie_k_steps > 0 else 0.0
        )

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

    @staticmethod
    def _project_l2_ball(delta: torch.Tensor, eps: float) -> torch.Tensor:
        """
        Project `delta` onto the L2 ball of radius `eps`, independently per batch element.

        Args:
            delta: (B, obs_dim)
            eps: radius

        Returns:
            projected delta: (B, obs_dim)
        """
        flat = delta.view(delta.size(0), -1)  # (B, D)
        norms = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)  # (B, 1)
        scale = (eps / norms).clamp(max=1.0)  # (B, 1)
        return (flat * scale).view_as(delta)

    # ERNIE methods
    def _ernie_adv_delta(
        self,
        actor_net: torch.nn.Module,
        obs: torch.Tensor,
        eps: float,
        k_steps: int,
        step_size: float,
        norm: Literal["linf", "l2"] = "linf",
    ) -> torch.Tensor:
        """
        ERNIE: inner maximization via PGD ascent to find delta that maximizes
        D(pi(o+delta), pi(o)) under ||delta|| <= eps.

        Args:
            actor_net: maps obs -> action. obs: (B, obs_dim) -> (B, act_dim)
            obs: (B, obs_dim)
            eps: perturbation budget
            k_steps: number of PGD steps
            step_size: PGD step size
            norm: "linf" or "l2"

        Returns:
            delta_adv: (B, obs_dim), detached
        """
        if eps <= 0.0 or k_steps <= 0 or step_size <= 0.0:
            return torch.zeros_like(obs)

        # Reference action (no gradients through this branch)
        with torch.no_grad():
            base_action = actor_net(obs)  # (B, act_dim)

        # Random init within constraint set
        delta = torch.empty_like(obs).uniform_(-eps, eps)
        if norm == "l2":
            delta = self._project_l2_ball(delta, eps)

        delta.requires_grad_(True)

        for _ in range(k_steps):
            pert_action = actor_net(obs + delta)

            # Maximize mean squared action deviation (stable for deterministic actors)
            objective = (pert_action - base_action).pow(2).mean()

            (grad,) = torch.autograd.grad(
                outputs=objective,
                inputs=delta,
                retain_graph=False,
                create_graph=False,
                only_inputs=True,
            )

            with torch.no_grad():
                if norm == "linf":
                    delta.add_(step_size * grad.sign())
                    delta.clamp_(-eps, eps)
                else:  # "l2"
                    delta.add_(step_size * grad)
                    delta.copy_(self._project_l2_ball(delta, eps))

            delta.requires_grad_(True)

        return delta.detach()

    # M3DDPG methods
    def _compute_adversarial_actions(
        self,
        unperturbed_agent_indices: list[int],
        actions: torch.Tensor,  # (batch, n_agents, act_dim)
        global_states: torch.Tensor,  # (batch, state_dim)
        critic: torch.nn.Module,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Return actions_adv where for j != agent_index:
            a_j_adv = a_j + eps_j
        and eps_j is a 1-step gradient move that *decreases* Q_i.
        """
        if self.m3_alpha == 0.0:
            # Degenerates to original MADDPG
            return actions.detach(), torch.zeros_like(actions)

        # Clone and mark for gradient wrt actions only
        actions_for_grad = actions.detach().clone().requires_grad_(True)
        batch_size = actions_for_grad.shape[0]

        # Flatten to feed critic
        joint_actions_flat = actions_for_grad.view(batch_size, -1)
        q_vals = critic(global_states, joint_actions_flat).mean()  # scalar

        # Gradient of Q wrt all actions
        (grad_actions,) = torch.autograd.grad(
            q_vals,
            actions_for_grad,
            retain_graph=False,
            create_graph=False,
        )
        # grad_actions: (batch, n_agents, act_dim)

        # Scale by |a_j| as in Eq.(17)
        act_norm = actions_for_grad.norm(dim=-1, keepdim=True)
        grad_norm = grad_actions.norm(dim=-1, keepdim=True) + 1e-8

        eps = -self.m3_alpha * act_norm * grad_actions / grad_norm

        # Zero perturbation for the current agent i
        mask = torch.ones_like(eps)
        for agent_index in unperturbed_agent_indices:
            mask[:, agent_index, :] = 0.0
        eps = eps * mask

        actions_adv = actions_for_grad + eps
        return actions_adv.detach(), eps.detach()  # no gradients through perturbation

    def _update_critic(
        self,
        learning_unit: DDPG,
        unperturbed_agent_indices: list[int],
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B, 1)
        next_global_states: torch.Tensor,
        next_actions: torch.Tensor,  # (B, N, act_dim) from target actors
        dones_i: torch.Tensor,
    ) -> dict[str, Any]:
        info: dict[str, Any] = {}

        # --- Step 1: build (possibly adversarial) next joint actions ---
        if self.use_m3:
            # M3DDPG: perturb OTHER agents' target actions for agent i
            next_actions_adv, eps = self._compute_adversarial_actions(
                unperturbed_agent_indices=unperturbed_agent_indices,
                actions=next_actions,  # (B, N, act_dim)
                global_states=next_global_states,  # (B, state_dim)
                critic=learning_unit.target_critic_net,  # target critic
            )
            next_joint_actions = next_actions_adv.view(next_actions_adv.size(0), -1)
        else:
            # Plain MADDPG
            next_joint_actions = next_actions.view(next_actions.size(0), -1)

        # --- Step 2: TD target ---
        with torch.no_grad():
            target_q = learning_unit.target_critic_net(
                next_global_states, next_joint_actions
            )
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values = learning_unit.critic_net(global_states, joint_actions)

        loss = F.mse_loss(q_values, q_target)

        learning_unit.critic_net_optimiser.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                learning_unit.critic_net.parameters(), self.max_grad_norm
            )

        learning_unit.critic_net_optimiser.step()

        with torch.no_grad():

            td = q_values - q_target

            # --- Value scale ---
            info["q_mean"] = q_values.mean().item()
            info["q_std"] = q_values.std(unbiased=False).item()

            info["q_target_mean"] = q_target.mean().item()
            info["q_target_std"] = q_target.std(unbiased=False).item()

            # --- TD error diagnostics ---
            td_abs = td.abs()
            info["td_abs_mean"] = td_abs.mean().item()
            info["td_abs_p95"] = td_abs.quantile(0.95).item()
            info["td_abs_max"] = td_abs.max().item()

            # --- Signed bias ---
            info["td_mean"] = td.mean().item()

            if self.use_m3:
                info["critic_m3_eps_norm_mean"] = eps.norm(dim=-1).mean().item()
                info["critic_m3_eps_norm_p95"] = eps.norm(dim=-1).quantile(0.95).item()

            # --- Critic loss ---
            info["critic_loss"] = loss.item()

        return info

    def _build_actor_contribution(
        self,
        actor_unit: DDPG,
        critic_unit: DDPG,
        controlled_agent_ids: list[str],
        unperturbed_agent_indices: list[int],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        replay_actions: torch.Tensor,  # (B, N, act_dim)
    ) -> tuple[
        list[torch.Tensor],  # actions_i_all
        torch.Tensor,  # actor_q_values
        torch.Tensor,  # actor_objective
        torch.Tensor,  # ernie_reg
        torch.Tensor | None,  # eps_norm_mean
    ]:
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
        ernie_regs = []

        for controlled_agent_id in controlled_agent_ids:
            agent_index = self.all_agent_ids.index(controlled_agent_id)

            obs_i = obs_tensors[controlled_agent_id]
            actions_i = actor_unit.actor_net(obs_i)

            actions_all[:, agent_index, :] = actions_i
            actions_i_all.append(actions_i)

            if self.use_ernie:
                delta_adv = self._ernie_adv_delta(
                    actor_net=actor_unit.actor_net,
                    obs=obs_i,
                    eps=self.ernie_eps,
                    k_steps=self.ernie_k_steps,
                    step_size=self.ernie_step_size,
                    norm=self.ernie_norm,
                )
                pred_action_adv = actor_unit.actor_net(obs_i + delta_adv)
                ernie_regs.append((pred_action_adv - actions_i).pow(2).mean())

        eps_norm_mean = None
        if self.use_m3:
            actions_adv, eps = self._compute_adversarial_actions(
                unperturbed_agent_indices=unperturbed_agent_indices,
                actions=actions_all,
                global_states=global_states,
                critic=critic_unit.critic_net,
            )

            for controlled_agent_id, actions_i in zip(
                controlled_agent_ids, actions_i_all
            ):
                agent_index = self.all_agent_ids.index(controlled_agent_id)
                actions_adv[:, agent_index, :] = actions_i

            actions_all = actions_adv
            eps_norm_mean = eps.norm(dim=-1).mean()

        joint_actions_flat = actions_all.reshape(batch_size, -1)

        with fnc.evaluating(critic_unit.critic_net):
            actor_q_values = critic_unit.critic_net(global_states, joint_actions_flat)

        actor_objective = -actor_q_values.mean()

        if self.use_ernie and ernie_regs:
            ernie_reg = torch.stack(ernie_regs).mean()
        else:
            ernie_reg = torch.tensor(0.0, device=global_states.device)

        return actions_i_all, actor_q_values, actor_objective, ernie_reg, eps_norm_mean

    def _update_actor(
        self,
        actor_unit: DDPG,
        critic_unit: DDPG,
        controlled_agent_ids: list[str],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        replay_actions: torch.Tensor,
    ) -> dict[str, Any]:

        info: dict[str, Any] = {}

        unperturbed_agent_indices = [
            self.all_agent_ids.index(agent_id) for agent_id in controlled_agent_ids
        ]

        actions_all: list[torch.Tensor] = []
        contribution_actions_all: list[list[torch.Tensor]] = []

        actor_q_values_all: list[torch.Tensor] = []
        actor_objectives_all: list[torch.Tensor] = []
        ernie_regs_all: list[torch.Tensor] = []
        m3_eps_norm_means_all: list[torch.Tensor] = []

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

        for contribution_agent_ids in contribution_groups:
            actions_i_all, actor_q_values, actor_objective, ernie_reg, eps_norm_mean = (
                self._build_actor_contribution(
                    actor_unit=actor_unit,
                    critic_unit=critic_unit,
                    controlled_agent_ids=contribution_agent_ids,
                    unperturbed_agent_indices=unperturbed_agent_indices,
                    obs_tensors=obs_tensors,
                    global_states=global_states,
                    replay_actions=replay_actions,
                )
            )

            actions_all.extend(actions_i_all)
            contribution_actions_all.append(actions_i_all)

            actor_q_values_all.append(actor_q_values)
            actor_objectives_all.append(actor_objective)

            if self.use_ernie:
                ernie_regs_all.append(ernie_reg)
            if self.use_m3 and eps_norm_mean is not None:
                m3_eps_norm_means_all.append(eps_norm_mean)

        actor_objective = torch.stack(actor_objectives_all).mean()

        actions_cat = torch.cat(actions_all, dim=0)
        actor_q_cat = torch.cat(actor_q_values_all, dim=0)

        reg = (actions_cat**2).mean() * 1e-3

        if self.use_ernie and ernie_regs_all:
            ernie_reg = torch.stack(ernie_regs_all).mean()
        else:
            ernie_reg = torch.tensor(0.0, device=global_states.device)

        actor_loss = actor_objective + reg + (self.ernie_lambda * ernie_reg)

        dq_da_values: list[torch.Tensor] = []

        for actor_q_values_i, contribution_actions in zip(
            actor_q_values_all,
            contribution_actions_all,
        ):
            for actions_i in contribution_actions:
                dq_da = torch.autograd.grad(
                    outputs=-actor_q_values_i.mean(),  # NOTE: uses Q-term only, excludes regularizers
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

            if self.use_ernie:
                info["ernie_reg"] = ernie_reg.item()

            if self.use_m3 and m3_eps_norm_means_all:
                m3_eps_norms_tensor = torch.stack(m3_eps_norm_means_all)
                info["actor_m3_eps_norm_mean"] = m3_eps_norms_tensor.mean().item()
                info["actor_m3_eps_norm_p95"] = m3_eps_norms_tensor.quantile(
                    0.95
                ).item()

        return info

    def _sample_training_batch(
        self, memory_buffer: MARLMemoryBuffer
    ) -> tuple[MADDPGBatch, dict[str, Any]]:
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
        next_actions = {}
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

        joint_actions = actions_tensor.reshape(actions_tensor.shape[0], -1)

        # ---------------------------------------------------------
        # Batch-level diagnostics once
        # ---------------------------------------------------------
        with torch.no_grad():
            info["joint_action_mean"] = actions_tensor.mean().item()
            info["joint_action_std"] = actions_tensor.std(unbiased=False).item()

            per_agent_abs_mean = actions_tensor.abs().mean(dim=(0, 2))
            per_agent_std = actions_tensor.std(dim=(0, 2), unbiased=False)

            info["replay_action_abs_mean"] = per_agent_abs_mean.mean().item()
            info["replay_action_abs_std_across_agents"] = per_agent_abs_mean.std(
                unbiased=False
            ).item()
            info["replay_action_std_mean"] = per_agent_std.mean().item()

            a_norm = actions_tensor / actions_tensor.norm(
                dim=2, keepdim=True
            ).clamp_min(1e-6)

            cos = torch.einsum("bna,bma->bnm", a_norm, a_norm)
            n = cos.shape[1]
            mask = ~torch.eye(n, device=cos.device, dtype=torch.bool)

            info["replay_action_cos_mean"] = cos[:, mask].mean().item()

            info["reward_mean"] = rewards_tensor.mean().item()
            info["done_frac"] = dones_tensor.float().mean().item()

        batch_data = MADDPGBatch(
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
            joint_actions=joint_actions,
        )

        return (batch_data, info)

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

        # ---------------------------------------------------------
        # Sample ONCE for all learning units
        # Shared minibatch: We draw one minibatch per training iteration and reuse it across updates.
        # This preserves an unbiased estimator of each update while reducing sampling-induced variance and
        # keeping joint transitions consistent for centralized critics.
        samples, batch_info = self._sample_training_batch(memory_buffer)
        info |= batch_info

        global_states = samples.global_states
        next_global_states = samples.next_global_states
        agent_states_tensors = samples.agent_states
        rewards_by_agent = samples.rewards_by_agent
        dones_by_agent = samples.dones_by_agent
        actions_tensor = samples.actions
        next_actions = samples.next_actions
        joint_actions = samples.joint_actions

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

            unperturbed_agent_indices = [
                self.all_agent_ids.index(agent_id) for agent_id in critic_agent_ids
            ]

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
                unperturbed_agent_indices=unperturbed_agent_indices,
                global_states=global_states,
                joint_actions=joint_actions,
                rewards_i=learning_unit_rewards,
                next_global_states=next_global_states,
                next_actions=next_actions,
                dones_i=learning_unit_dones,
            )

            info.update({f"{critic_id}_{k}": v for k, v in critic_info.items()})

        # ---------------------------------------------------------
        # Actor updates are grouped by ACTOR ownership, not critic ownership.
        #
        # individual:
        #     one actor update per env agent
        #
        # team_critic:
        #     one actor update per env agent
        #     but critics are shared at the team level
        #
        # team_all:
        #     one shared actor update per team
        # ---------------------------------------------------------
        for actor_id, actor_agent_ids in self.actor_id_to_agent_ids.items():
            actor_unit = self.learning_units[actor_id]

            # All agents controlled by the same actor learning unit are guaranteed
            # to map to the same critic learning unit.
            #
            # Therefore we can use the first controlled agent to recover the critic ID.
            #
            # individual:
            #   actor_agent_ids = [agent_0]
            #   critic_id = agent_0
            #
            # team_critic:
            #   actor_agent_ids = [agent_0]
            #   critic_id = adversary
            #
            # team_all:
            #   actor_agent_ids = [agent_0, agent_1, agent_2]
            #   critic_id = adversary

            critic_id = self.agent_id_to_critic_id[actor_agent_ids[0]]
            critic_unit = self.learning_units[critic_id]

            actor_info = self._update_actor(
                actor_unit=actor_unit,
                critic_unit=critic_unit,
                controlled_agent_ids=actor_agent_ids,
                obs_tensors=agent_states_tensors,
                global_states=global_states,
                replay_actions=actions_tensor,
            )

            info.update({f"{actor_id}_{k}": v for k, v in actor_info.items()})

        # ---------------------------------------------------------
        # Target network updates
        # ---------------------------------------------------------
        for learning_unit in self.learning_units.values():
            learning_unit.update_target_networks()

        # ---------------------------------------------------------
        # Cross-learning-unit diagnostics
        # ---------------------------------------------------------
        metrics = list(critic_info.keys()) + list(actor_info.keys())
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
            learning_unit_filepath = os.path.join(filepath, f"{learning_unit_id}")
            learning_unit_filename = f"{filename}_{learning_unit_id}_checkpoint"
            learning_unit.save_models(learning_unit_filepath, learning_unit_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for learning_unit_id, learning_unit in self.learning_units.items():
            learning_unit_filepath = os.path.join(filepath, f"{learning_unit_id}")
            learning_unit_filename = f"{filename}_{learning_unit_id}_checkpoint"
            learning_unit.load_models(learning_unit_filepath, learning_unit_filename)

        logging.info("models and optimisers have been loaded...")
