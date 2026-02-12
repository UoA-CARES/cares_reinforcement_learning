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
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy.SAC import SAC
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.util.configurations import MASACConfig


class MASAC(Algorithm[MARLObservation, list[np.ndarray], MARLMemoryBuffer]):
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
        agent: SAC,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B,) or (B, 1)
        next_global_states: torch.Tensor,
        next_joint_actions: torch.Tensor,  # (B, N * act_dim) from target policies (SAMPLED)
        next_logp_i: torch.Tensor,  # (B, 1) log pi_i(a_i' | o_i') for NEXT state
        dones_i: torch.Tensor,  # (B,) or (B, 1)
    ):

        # ---- Step 1: TD target with entropy term ----
        with torch.no_grad():
            target_q1, target_q2 = agent.target_critic_net(
                next_global_states, next_joint_actions
            )
            target_q = torch.min(target_q1, target_q2)

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

        return {
            "critic_loss_one": critic_loss_one.item(),
            "critic_loss_two": critic_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

    def _update_actor_alpha(
        self,
        agent: SAC,
        agent_index: int,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        current_actions_tensor: torch.Tensor,  # (B, N, act_dim) sampled under no_grad
    ):
        agent_ids = list(obs_tensors.keys())
        batch_size = global_states.shape[0]

        actions_all = current_actions_tensor.clone()  # no graphs carried

        # ---------------------------------------------------------
        # Sample CURRENT actions for all agents (detach others)
        # ---------------------------------------------------------
        obs_i = obs_tensors[agent_ids[agent_index]]
        action_i, logp_i, _ = agent.actor_net(obs_i)  # grads for i only

        actions_all[:, agent_index, :] = action_i  # only i is live

        joint_actions_flat = actions_all.reshape(batch_size, -1)

        # ---------------------------------------------------------
        # Step 4: Compute actor loss: -Q_i(x, a_1,...,a_i,...,a_N)
        # ---------------------------------------------------------
        q_val_one, q_val_two = agent.critic_net(global_states, joint_actions_flat)

        q_val = torch.min(q_val_one, q_val_two)

        actor_loss = (agent.alpha * logp_i - q_val).mean()

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
            agent.log_alpha * (logp_i + agent.target_entropy).detach()
        ).mean()

        agent.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        agent.log_alpha_optimizer.step()

        return {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
            "alpha": agent.alpha.item(),
        }

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
        next_actions = []
        next_logps = []

        for agent, agent_id in zip(self.agent_networks, agent_ids):
            obs_next = next_agent_states[agent_id]

            # You need a method like: action, logp = agent.actor_net.sample(obs_next)
            # where action is already in env action space (or consistently scaled)
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
        if self.learn_counter % self.policy_update_freq == 0:
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
