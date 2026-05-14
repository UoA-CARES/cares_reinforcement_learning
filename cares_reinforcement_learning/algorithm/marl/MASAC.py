"""
MASAC implementation overview
-----------------------------

MASAC extends SAC to the multi-agent setting using:

    - centralized twin critics,
    - decentralized stochastic actors,
    - entropy-regularized policy optimization,
    - and optional team-based parameter sharing.

Unlike MADDPG/MATD3, MASAC optimizes stochastic policies under the current
joint policy distribution. This changes how actor updates are constructed in
team-sharing settings.

This implementation supports two main parameter-sharing modes:

    sharing_mode:
        "individual" -> one SAC learning unit per environment agent
        "team"       -> one shared SAC learning unit per environment team

Terminology
-----------

Environment agent:
    An agent that exists in the environment, e.g.
        adversary_0, adversary_1, agent_0

Learning unit:
    A trainable SAC bundle containing:
        actor, twin critics, target critics, entropy tuning, and optimisers.

    In individual mode:
        each environment agent has its own learning unit.

    In team mode:
        all agents in the same team share one learning unit.

Controlled agents:
    The environment agents assigned to a learning unit.

Centralized training, decentralized execution
---------------------------------------------

MASAC follows the CTDE (centralized training, decentralized execution)
paradigm.

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

Soft Actor-Critic extensions
----------------------------

Twin critics:
    MASAC uses two critics per learning unit and minimizes the target over both
    critics:

        target_q = min(Q1_target, Q2_target)

    This reduces overestimation bias.

Entropy regularization:
    MASAC optimizes a maximum-entropy objective:

        J_pi = E[alpha * log_pi(a|s) - Q(s, a)]

    This encourages stochastic exploration and improves robustness.

Automatic entropy tuning:
    The entropy temperature alpha is learned automatically using:

        target_entropy

    This adapts exploration pressure during training.

Stochastic policies:
    Actors output:
        - sampled actions,
        - action log probabilities,
        - and latent distribution parameters.

    Actions are sampled using the reparameterization trick for low-variance
    policy gradients.

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

    1. Sample NEXT actions from current target policies.
    2. Build the NEXT joint action from sampled stochastic actions.
    3. Compute the soft target:

            y = r_i + gamma * (1 - done_i)
                * (min(Q1_target, Q2_target) - alpha * log_pi_i)

    4. Regress both critics against the replay-buffer joint action:

            Q1_i(s, a_replay_joint)
            Q2_i(s, a_replay_joint)

In individual mode:
    r_i and done_i are the controlled agent's own reward/done.

In team mode:
    r_i and done_i are aggregated across the controlled agents.

Actor update
------------

Unlike deterministic methods such as MADDPG/MATD3, MASAC optimizes an
expectation under the CURRENT stochastic joint policy distribution.

For actor optimisation:

    1. Current stochastic actions are sampled for ALL environment agents.
    2. Controlled agents' sampled actions are inserted into the joint action.
    3. The centralized critic evaluates the resulting joint action.
    4. Gradients flow only through the controlled agents' sampled actions.

This answers the question:

    "Under the current stochastic joint policy, would changing the actions of
    the agents controlled by this learning unit increase the expected soft
    value?"

Replay actions are NOT used during actor optimisation because SAC's objective
is defined under the current policy distribution rather than replay behaviour.

Entropy aggregation in team mode
--------------------------------

When using team sharing:

    - team rewards are aggregated using the mean reward,
    - and team entropy contributions are aggregated using the mean log-probability.

This keeps the entropy and value scales approximately aligned with the team
critic objective.

Practical notes
---------------

For cooperative environments such as simple_spread:

    individual:
        closest to standard MASAC, with one stochastic policy per environment
        agent.

    team:
        often improves coordination consistency because all controlled agents
        share:
            - one actor,
            - one entropy objective,
            - and one centralized team critic.

Compared to MADDPG/MATD3, MASAC is typically:
    - more exploratory,
    - more robust to noisy environments,
    - less sensitive to local optima,
    - but potentially less sample efficient due to stochastic policies and
      entropy regularization.
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
from cares_reinforcement_learning.algorithm.configurations import MASACConfig
from cares_reinforcement_learning.algorithm.policy.SAC import SAC
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.networks import functional as fnc
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


@dataclass(frozen=True, slots=True)
class MASACBatch:
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
    next_logps: torch.Tensor  # (B, N, 1)
    next_actions_by_agent: dict[str, torch.Tensor]
    next_logps_by_agent: dict[str, torch.Tensor]
    joint_actions: torch.Tensor  # (B, N * A)
    next_joint_actions: torch.Tensor  # (B, N * A)


class MASAC(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        learning_units: dict[str, SAC],
        all_agent_ids: list[str],
        agent_id_to_learning_unit_id: dict[str, str],
        learning_unit_to_agent_ids: dict[str, list[str]],
        config: MASACConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.learning_units = learning_units

        # Shared Actor/Critic per team or individual Actor/Critic per agent
        self.sharing_mode = config.sharing_mode

        # All environment agent IDs and counts, e.g.
        # ["adversary_0", "adversary_1", "adversary_2", "agent_0"]
        self.all_agent_ids = all_agent_ids
        self.num_agents = len(all_agent_ids)

        # Maps env agent -> learning unit.
        # individual:
        #   adversary_0 -> adversary_0
        # team:
        #   adversary_0 -> adversary
        self.agent_id_to_learning_unit_id = agent_id_to_learning_unit_id

        # Maps learning unit -> env agents controlled by it.
        # individual:
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
        self.target_update_freq = config.target_update_freq

        self.max_grad_norm = config.max_grad_norm

        self.learn_counter = 0

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
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
        agent: SAC,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B,) or (B, 1)
        next_global_states: torch.Tensor,
        next_joint_actions: torch.Tensor,  # (B, N * act_dim) from target policies (SAMPLED)
        next_logp_i: torch.Tensor,  # (B, 1) log pi_i(a_i' | o_i') for NEXT state
        dones_i: torch.Tensor,  # (B,) or (B, 1)
    ):
        info: dict[str, Any] = {}

        # ---- Step 1: TD target with entropy term ----
        with torch.no_grad():
            target_q_values_one, target_q_values_two = agent.target_critic_net(
                next_global_states, next_joint_actions
            )
            target_q = torch.min(target_q_values_one, target_q_values_two)

            q_target = rewards_i + self.gamma * (1.0 - dones_i) * (
                target_q - agent.alpha * next_logp_i
            )

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

        # ---------------------------------------------------------
        # Step 3: diagnostics (collated at bottom)
        # ---------------------------------------------------------
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

            # --- Soft target decomposition (SAC-specific) ---
            # min_target_q_mean: the conservative bootstrap value from twin critics (pre-entropy)
            # entropy_term_mean: magnitude of entropy regularization in the target (alpha * log_pi is usually negative)
            # soft_target_value_mean: the exact term used inside the Bellman target before reward/discount
            min_target_q = torch.minimum(target_q_values_one, target_q_values_two)

            # alpha_log_pi is typically negative; entropy_bonus is typically positive
            alpha_log_pi = agent.alpha * next_logp_i
            # this is what gets ADDED to minQ in the target
            entropy_bonus = -agent.alpha * next_logp_i

            soft_target_value = min_target_q + entropy_bonus  # == minQ - alpha*log_pi

            info["target_min_q_mean"] = min_target_q.mean().item()
            info["alpha_log_pi_mean"] = alpha_log_pi.mean().item()
            info["entropy_bonus_mean"] = entropy_bonus.mean().item()
            info["soft_target_value_mean"] = soft_target_value.mean().item()

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

    def _build_actor_alpha_contribution(
        self,
        learning_unit: SAC,
        controlled_agent_ids: list[str],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        current_actions_tensor: torch.Tensor,
    ) -> tuple[
        list[torch.Tensor],
        list[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Build the MASAC actor/alpha contribution for one learning unit.

        individual sharing:
            controlled_agent_ids contains one environment agent.

        team sharing:
            controlled_agent_ids contains all agents controlled by the shared team
            actor, and all of their current policy actions are inserted into the
            joint action together.

        This gives MASAC the natural SAC-style objective under the current joint
        policy, rather than a MADDPG-style one-action-at-a-time counterfactual.
        """
        batch_size = global_states.shape[0]
        actions_all = current_actions_tensor.clone()

        pi_all: list[torch.Tensor] = []
        log_pi_all: list[torch.Tensor] = []

        for controlled_agent_id in controlled_agent_ids:
            agent_index = self.all_agent_ids.index(controlled_agent_id)

            obs_i = obs_tensors[controlled_agent_id]
            pi_i, log_pi_i, _ = learning_unit.actor_net(obs_i)

            actions_all[:, agent_index, :] = pi_i

            pi_all.append(pi_i)
            log_pi_all.append(log_pi_i)

        joint_actions_flat = actions_all.reshape(batch_size, -1)

        with fnc.evaluating(learning_unit.critic_net):
            qf_pi_one, qf_pi_two = learning_unit.critic_net(
                global_states,
                joint_actions_flat,
            )

        min_qf_pi = torch.min(qf_pi_one, qf_pi_two)

        # For a team-shared actor, aggregate the entropy contribution across the
        # controlled agents while keeping the reward/Q scale comparable to the
        # team critic target, which also uses mean team reward.
        log_pi = torch.stack(log_pi_all, dim=1).mean(dim=1)

        actor_loss = (learning_unit.alpha * log_pi - min_qf_pi).mean()

        return pi_all, log_pi_all, min_qf_pi, qf_pi_one, qf_pi_two, actor_loss

    def _update_actor_alpha(
        self,
        learning_unit: SAC,
        controlled_agent_ids: list[str],
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        current_actions_tensor: torch.Tensor,
    ) -> dict[str, Any]:
        """
        MASAC actor + alpha update.

        SAC optimises an expectation under the current stochastic policy. Therefore,
        when a learning unit controls multiple agents in team sharing mode, all
        controlled agents' current policy actions are evaluated together under one
        shared actor objective.

        evaluate all controlled agents' new actions together
            one team loss

            actor_loss = -Q_team(s, [π(o_1), π(o_2), π(o_3)]).mean()
        """
        info: dict[str, Any] = {}

        (
            pi_all,
            log_pi_all,
            min_qf_pi,
            qf_pi_one,
            qf_pi_two,
            actor_loss,
        ) = self._build_actor_alpha_contribution(
            learning_unit=learning_unit,
            controlled_agent_ids=controlled_agent_ids,
            obs_tensors=obs_tensors,
            global_states=global_states,
            current_actions_tensor=current_actions_tensor,
        )

        pi_cat = torch.cat(pi_all, dim=0)
        log_pi_cat = torch.cat(log_pi_all, dim=0)

        dq_da_values: list[torch.Tensor] = []

        for pi_i in pi_all:
            dq_da = torch.autograd.grad(
                outputs=actor_loss,
                inputs=pi_i,
                retain_graph=True,
                create_graph=False,
                allow_unused=False,
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

        alpha_loss = -(
            learning_unit.log_alpha
            * (log_pi_cat + learning_unit.target_entropy).detach()
        ).mean()

        learning_unit.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        learning_unit.log_alpha_optimizer.step()

        with torch.no_grad():
            info["dq_da_abs_mean"] = dq_da_cat.abs().mean().item()
            info["dq_da_norm_mean"] = dq_da_cat.norm(dim=1).mean().item()
            info["dq_da_norm_p95"] = dq_da_cat.norm(dim=1).quantile(0.95).item()

            info["log_pi_mean"] = log_pi_cat.mean().item()
            info["log_pi_std"] = log_pi_cat.std(unbiased=False).item()

            info["pi_action_abs_mean"] = pi_cat.abs().mean().item()
            info["pi_action_std"] = pi_cat.std(unbiased=False).item()
            info["pi_action_saturation_frac"] = (
                (pi_cat.abs() > 0.95).float().mean().item()
            )

            info["min_qf_pi_mean"] = min_qf_pi.mean().item()
            info["qf_pi_gap_abs_mean"] = (qf_pi_one - qf_pi_two).abs().mean().item()

            entropy_gap = -(log_pi_cat + learning_unit.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = learning_unit.alpha.item()
            info["log_alpha"] = learning_unit.log_alpha.item()

        return info

    def _sample_training_batch(
        self,
        memory_buffer: MARLMemoryBuffer,
    ) -> tuple[MASACBatch, dict[str, Any]]:
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

        joint_actions = actions_tensor.reshape(actions_tensor.shape[0], -1)

        next_actions_by_agent: dict[str, torch.Tensor] = {}
        next_logps_by_agent: dict[str, torch.Tensor] = {}

        for agent_id in self.all_agent_ids:
            learning_unit_id = self.agent_id_to_learning_unit_id[agent_id]
            learning_unit = self.learning_units[learning_unit_id]

            obs_next_i = sample_tensor.next_observation.agent_states[agent_id]

            with torch.no_grad():
                with fnc.evaluating(learning_unit.actor_net):
                    next_action_i, next_logp_i, _ = learning_unit.actor_net(obs_next_i)

            next_actions_by_agent[agent_id] = next_action_i
            next_logps_by_agent[agent_id] = next_logp_i

        next_actions_tensor = torch.stack(
            [next_actions_by_agent[a] for a in self.all_agent_ids],
            dim=1,
        )

        next_logps_tensor = torch.stack(
            [next_logps_by_agent[a] for a in self.all_agent_ids],
            dim=1,
        )

        next_joint_actions = next_actions_tensor.reshape(
            next_actions_tensor.shape[0], -1
        )

        with torch.no_grad():
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

            info["next_logp_mean_all_agents"] = next_logps_tensor.mean().item()
            info["next_logp_std_all_agents"] = next_logps_tensor.std(
                unbiased=False
            ).item()
            info["next_entropy_mean_all_agents"] = (-next_logps_tensor).mean().item()

            info["next_action_abs_mean_all_agents"] = (
                next_actions_tensor.abs().mean().item()
            )
            info["next_action_std_all_agents"] = next_actions_tensor.std(
                unbiased=False
            ).item()
            info["next_action_saturation_frac_all_agents"] = (
                (next_actions_tensor.abs() > 0.95).float().mean().item()
            )

            info["reward_mean"] = rewards_tensor.mean().item()
            info["done_frac"] = dones_tensor.float().mean().item()

        batch_data = MASACBatch(
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
            next_logps=next_logps_tensor,
            next_actions_by_agent=next_actions_by_agent,
            next_logps_by_agent=next_logps_by_agent,
            joint_actions=joint_actions,
            next_joint_actions=next_joint_actions,
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
        next_logps_by_agent = samples.next_logps_by_agent
        joint_actions = samples.joint_actions
        next_joint_actions = samples.next_joint_actions

        # ---------------------------------------------------------
        # Critic update
        # ---------------------------------------------------------
        for learning_unit_id, learning_unit in self.learning_units.items():
            controlled_agent_ids = self.learning_unit_to_agent_ids[learning_unit_id]

            # ---------------------------------------------------------
            # Build learning-unit rewards/dones from controlled agents
            # individual: direct per-agent tensors
            # team: aggregate across controlled agents
            # ---------------------------------------------------------
            if self.sharing_mode == "individual":
                controlled_agent_id = controlled_agent_ids[0]
                learning_unit_rewards = rewards_by_agent[controlled_agent_id]
                learning_unit_dones = dones_by_agent[controlled_agent_id]
                next_logp_i = next_logps_by_agent[controlled_agent_id]  # (B, 1)
            elif self.sharing_mode == "team":
                learning_unit_rewards = torch.stack(
                    [rewards_by_agent[a] for a in controlled_agent_ids],
                    dim=1,
                ).mean(dim=1)
                learning_unit_dones = torch.stack(
                    [dones_by_agent[a] for a in controlled_agent_ids],
                    dim=1,
                ).amax(dim=1)
                # Team/shared actor: aggregate entropy contribution across controlled agents
                next_logp_i = torch.stack(
                    [next_logps_by_agent[a] for a in controlled_agent_ids],
                    dim=1,
                ).mean(dim=1)
            else:
                raise ValueError(f"Invalid sharing_mode: {self.sharing_mode}")

            with torch.no_grad():
                info[f"{learning_unit_id}_reward_mean"] = (
                    learning_unit_rewards.mean().item()
                )
                info[f"{learning_unit_id}_done_frac"] = (
                    learning_unit_dones.float().mean().item()
                )
                info[f"{learning_unit_id}_next_logp_mean"] = next_logp_i.mean().item()

            critic_info = self._update_critic(
                agent=learning_unit,
                global_states=global_states,
                joint_actions=joint_actions,  # from replay at time t
                rewards_i=learning_unit_rewards,
                next_global_states=next_global_states,
                next_joint_actions=next_joint_actions,  # sampled at t+1
                next_logp_i=next_logp_i,
                dones_i=learning_unit_dones,
            )

            for key, value in critic_info.items():
                info[f"{learning_unit_id}_{key}"] = value

        # ---------------------------------------------------------
        # ACTOR + ALPHA UPDATES — usually every step in SAC
        # ---------------------------------------------------------
        update_actor = self.learn_counter % self.policy_update_freq == 0
        if update_actor:
            # ---------------------------------------------------------
            # For MASAC, sample current actions from all agents when
            # computing each agent’s actor loss.
            # Other agents’ actions are treated as fixed (no gradients).
            # This approximates the expectation over joint actions under
            # the current stochastic policy.
            # ---------------------------------------------------------
            with torch.no_grad():
                current_actions_list = []

                for agent_id in self.all_agent_ids:
                    learning_unit_id = self.agent_id_to_learning_unit_id[agent_id]
                    learning_unit = self.learning_units[learning_unit_id]

                    obs_j = agent_states[agent_id]
                    action_j, _, _ = learning_unit.actor_net(obs_j)

                    current_actions_list.append(action_j)

                current_actions_tensor = torch.stack(current_actions_list, dim=1)

            for learning_unit_id, learning_unit in self.learning_units.items():
                controlled_agent_ids = self.learning_unit_to_agent_ids[learning_unit_id]

                actor_info = self._update_actor_alpha(
                    learning_unit=learning_unit,
                    controlled_agent_ids=controlled_agent_ids,
                    obs_tensors=agent_states,
                    global_states=global_states,
                    current_actions_tensor=current_actions_tensor,
                )
                for key, value in actor_info.items():
                    info[f"{learning_unit_id}_{key}"] = value

        # ---------------------------------------------------------
        # Target critic updates (Polyak) — usually every step in SAC
        # ---------------------------------------------------------
        if self.learn_counter % self.target_update_freq == 0:
            for learning_unit in self.learning_units.values():
                learning_unit.update_target_networks()

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

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for learning_unit_id, learning_unit in self.learning_units.items():
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_agent_{learning_unit_id}_checkpoint"
            learning_unit.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for learning_unit_id, learning_unit in self.learning_units.items():
            agent_filepath = os.path.join(filepath, f"{learning_unit_id}")
            agent_filename = f"{filename}_agent_{learning_unit_id}_checkpoint"
            learning_unit.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
