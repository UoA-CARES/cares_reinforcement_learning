"""
MASAC (Multi-Agent Soft Actor-Critic) implementation notes
----------------------------------------------------------

This algorithm extends SAC to the multi-agent setting using centralized critics
and decentralized stochastic actors.

Replay sampling:
- A single minibatch is sampled per training iteration and reused across agents.
- This provides an unbiased estimate of each agent's update while reducing
  sampling-induced variance and ensuring consistency of joint transitions.

Critic updates:
- Next-state actions are sampled from the current stochastic policies.
- Target critics are used to compute TD targets including the entropy term.
- Each agent's entropy contribution is handled independently.

Actor updates:
- Policies are stochastic and optimized under a maximum-entropy objective.
- For actor updates, current actions are sampled for ALL agents.
- When updating agent i, gradients flow only through agent i's action;
  other agents' actions are detached.
- This aligns the update with the expectation under the current joint policy
  distribution, which is required by SAC's objective.

Rationale:
- Unlike deterministic methods (MADDPG/MATD3), SAC optimizes an expectation
  over actions drawn from the current policy.
- Using replay actions for other agents would evaluate Q under a stale joint
  behavior distribution, introducing additional bias as policies evolve.
"""

import logging
import os
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


class MASAC(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        agents: list[SAC],
        config: MASACConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

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

        agent_ids = list(agent_states.keys())
        actions = {}

        for i, agent in enumerate(self.agent_networks):
            agent_name = agent_ids[i]  # consistent ordering in dict
            obs_i = agent_states[agent_name]
            avail_i = avail_actions[agent_name]

            agent_observation = SARLObservation(
                vector_state=obs_i,
                available_actions=avail_i,
            )

            agent_sample = agent.act(agent_observation, evaluation)
            actions[agent_name] = agent_sample.action

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

    def _update_actor_alpha(
        self,
        agent: SAC,
        agent_index: int,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        current_actions_tensor: torch.Tensor,  # (B, N, act_dim) sampled under no_grad
    ):
        info: dict[str, Any] = {}

        agent_ids = list(obs_tensors.keys())
        batch_size = global_states.shape[0]

        actions_all = current_actions_tensor.clone()  # no graphs carried

        # ---------------------------------------------------------
        # Sample CURRENT actions for all agents (detach others)
        # ---------------------------------------------------------
        obs_i = obs_tensors[agent_ids[agent_index]]
        pi_i, log_pi_i, _ = agent.actor_net(obs_i)  # grads for i only

        actions_all[:, agent_index, :] = pi_i  # only i is live

        joint_actions_flat = actions_all.reshape(batch_size, -1)

        # ---------------------------------------------------------
        # Step 4: Compute actor loss: -Q_i(x, a_1,...,a_i,...,a_N)
        # ---------------------------------------------------------
        with fnc.evaluating(agent.critic_net):
            qf_pi_one, q_pi_two = agent.critic_net(global_states, joint_actions_flat)

        min_qf_pi = torch.min(qf_pi_one, q_pi_two)

        actor_loss = (agent.alpha * log_pi_i - min_qf_pi).mean()

        # ---------------------------------------------------------
        # Stochastic Policy Gradient Strength (∇a [α log π(a|s) − Q(s,a)])
        # ---------------------------------------------------------
        # Measures how steep the entropy-regularized critic objective is
        # w.r.t. the sampled policy actions.
        #
        # ~0 early  -> critic surface and entropy term nearly flat;
        #              actor receives weak learning signal.
        #
        # Very large -> critic or entropy term is very sharp around policy
        #               actions; can lead to unstable or overly aggressive
        #               actor updates.
        dq_da = torch.autograd.grad(
            outputs=actor_loss,
            inputs=pi_i,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]

        with torch.no_grad():
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

        # ---------------------------------------------------------
        # Step 6: Alpha loss and update
        # ---------------------------------------------------------
        alpha_loss = -(
            agent.log_alpha * (log_pi_i + agent.target_entropy).detach()
        ).mean()

        agent.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        agent.log_alpha_optimizer.step()

        with torch.no_grad():
            # --- Policy entropy diagnostics (exploration health) ---
            # log_pi more negative -> higher entropy (more stochastic). Less negative -> lower entropy (more deterministic).
            info["log_pi_mean"] = log_pi_i.mean().item()
            info["log_pi_std"] = log_pi_i.std().item()

            # --- Action magnitude/saturation (tanh policies) ---
            # High saturation fraction can indicate the policy is slamming bounds; may reduce effective gradients.
            info["pi_action_abs_mean"] = pi_i.abs().mean().item()
            info["pi_action_std"] = pi_i.std().item()
            info["pi_action_saturation_frac"] = (
                (pi_i.abs() > 0.95).float().mean().item()
            )

            # --- On-policy critic signal ---
            # min_qf_pi_mean should generally increase as the policy improves (higher value actions under the policy).
            info["min_qf_pi_mean"] = min_qf_pi.mean().item()

            # --- Twin critics disagreement at policy actions (more relevant than replay actions) ---
            # Large gap here means critics disagree on what the current policy is doing (can destabilize actor updates).
            info["qf_pi_gap_abs_mean"] = (qf_pi_one - q_pi_two).abs().mean().item()

            # --- Entropy gap (alpha tuning health) ---
            # entropy_gap ~ 0 means entropy matches target.
            # > 0: entropy too low -> alpha should increase; < 0: entropy too high -> alpha should decrease.
            entropy_gap = -(log_pi_i + agent.target_entropy)
            info["entropy_gap_mean"] = entropy_gap.mean().item()

            # --- Losses and temperature ---
            info["actor_loss"] = actor_loss.item()
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = agent.alpha.item()
            info["log_alpha"] = agent.log_alpha.item()

        return info

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
        # Build NEXT actions by sampling from CURRENT policies (not target)
        # ---------------------------------------------------------
        # In SAC, targets use a' ~ pi(·|o') (reparameterized), then evaluate target critics.
        # Computing next_joint_actions once outside ensures every agent sees the same bootstrapping sample for that minibatch.
        next_actions = []
        next_logps = []

        for agent, agent_id in zip(self.agent_networks, agent_ids):
            obs_next = next_agent_states[agent_id]

            with torch.no_grad():
                with fnc.evaluating(agent.actor_net):
                    next_action_j, next_logp_j, _ = agent.actor_net(
                        obs_next
                    )  # (B, act_dim), (B, 1)

            next_actions.append(next_action_j)
            next_logps.append(next_logp_j)

        # (B, N, act_dim) and (B, N, 1)
        next_actions_tensor = torch.stack(next_actions, dim=1)
        next_logps_tensor = torch.stack(next_logps, dim=1)

        next_joint_actions = next_actions_tensor.view(batch_size, -1)

        # ---------------------------------------------------------
        # CRITIC UPDATES (every step)
        # ---------------------------------------------------------
        for agent_index, agent in enumerate(self.agent_networks):
            rewards_i = rewards_tensor[:, agent_index]
            dones_i = dones_tensor[:, agent_index]

            # For MASAC, entropy term typically uses ONLY agent i's logp (common)
            next_logp_i = next_logps_tensor[:, agent_index, :]  # (B, 1)

            critic_info = self._update_critic(
                agent=agent,
                global_states=global_states,
                joint_actions=joint_actions,  # from replay at time t
                rewards_i=rewards_i,
                next_global_states=next_global_states,
                next_joint_actions=next_joint_actions,  # sampled at t+1
                next_logp_i=next_logp_i,
                dones_i=dones_i,
            )

            for key, value in critic_info.items():
                info[f"agent_{agent_index}_{key}"] = value

        # ---------------------------------------------------------
        # ACTOR + ALPHA UPDATES — usually every step in SAC
        # ---------------------------------------------------------
        update_actor = self.learn_counter % self.policy_update_freq == 0
        if update_actor:
            # ---------------------------------------------------------
            # For MASAC, we sample current actions from all agents when
            # computing each agent’s actor loss, detaching other agents’ samples.
            # This aligns the update with SAC’s maximum-entropy objective,
            # which is an expectation over actions drawn from the current stochastic policy.
            # ---------------------------------------------------------
            with torch.no_grad():
                current_actions = []
                for agent_j, agent_id_j in zip(self.agent_networks, agent_ids):
                    obs_j = agent_states[agent_id_j]
                    a_j, _, _ = agent_j.actor_net(obs_j)
                    current_actions.append(a_j)
                # (B, N, act_dim)
                current_actions_tensor = torch.stack(current_actions, dim=1)

            for agent_index, agent in enumerate(self.agent_networks):
                actor_info = self._update_actor_alpha(
                    agent=agent,
                    agent_index=agent_index,
                    obs_tensors=agent_states,
                    global_states=global_states,
                    current_actions_tensor=current_actions_tensor,
                )
                for key, value in actor_info.items():
                    info[f"agent_{agent_index}_{key}"] = value

        # ---------------------------------------------------------
        # Target critic updates (Polyak) — usually every step in SAC
        # ---------------------------------------------------------
        if self.learn_counter % self.target_update_freq == 0:
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

            # --- Next-policy sampling health (SAC target actions) ---
            # These catch entropy collapse and alpha/logp pathologies early
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
