"""
Original Paper: https://arxiv.org/pdf/1706.02275

Original Code (TensorFlow): https://github.com/openai/maddpg/tree/master
"""

import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)
from cares_reinforcement_learning.util.configurations import MADDPGConfig


class MADDPG(Algorithm[MARLObservation, MARLMemoryBuffer]):
    def __init__(
        self,
        agents: list[DDPG],
        config: MADDPGConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agents
        self.num_agents = len(agents)

        self.gamma = config.gamma
        self.tau = config.tau

        self.max_grad_norm = config.max_grad_norm

        self.alpha = config.alpha  # adversarial perturbation scale

        self.learn_counter = 0

    # TODO verify that the ordering of agents is consistent
    def select_action_from_policy(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> list[np.ndarray]:
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

            action = agent.select_action_from_policy(agent_observation, evaluation)
            actions.append(action)

        return actions

    def _compute_adversarial_actions(
        self,
        agent_index: int,
        actions: torch.Tensor,  # (batch, n_agents, act_dim)
        global_states: torch.Tensor,  # (batch, state_dim)
        critic: torch.nn.Module,
    ) -> torch.Tensor:
        """
        Return actions_adv where for j != agent_index:
            a_j_adv = a_j + eps_j
        and eps_j is a 1-step gradient move that *decreases* Q_i.
        """
        if self.alpha == 0.0:
            # Degenerates to original MADDPG
            return actions

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

        eps = -self.alpha * act_norm * grad_actions / grad_norm

        # Zero perturbation for the current agent i
        mask = torch.ones_like(eps)
        mask[:, agent_index, :] = 0.0
        eps = eps * mask

        actions_adv = actions_for_grad + eps
        return actions_adv.detach()  # no gradients through perturbation

    def _update_critic(
        self,
        agent: DDPG,
        agent_index: int,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,  # (B, N * act_dim) from replay
        rewards_i: torch.Tensor,  # (B, 1)
        next_global_states: torch.Tensor,
        next_actions_tensor: torch.Tensor,  # (B, N, act_dim) from target actors
        dones_i: torch.Tensor,
    ):
        # --- Step 1: build (possibly adversarial) next joint actions ---
        if self.alpha != 0.0:
            # M3DDPG: perturb OTHER agents' target actions for agent i
            next_actions_adv = self._compute_adversarial_actions(
                agent_index=agent_index,
                actions=next_actions_tensor,  # (B, N, act_dim)
                global_states=next_global_states,  # (B, state_dim)
                critic=agent.target_critic_net,  # target critic
            )
            next_joint_actions = next_actions_adv.view(next_actions_adv.size(0), -1)
        else:
            # Plain MADDPG
            next_joint_actions = next_actions_tensor.view(
                next_actions_tensor.size(0), -1
            )

        # --- Step 2: TD target ---
        with torch.no_grad():
            target_q = agent.target_critic_net(next_global_states, next_joint_actions)
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        print(q_target.shape, q_target.mean())

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values = agent.critic_net(global_states, joint_actions)

        print(q_values.shape, q_values.mean())

        loss = F.mse_loss(q_values, q_target)

        print(loss)

        agent.critic_net_optimiser.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.critic_net.parameters(), self.max_grad_norm
            )

        agent.critic_net_optimiser.step()

        return {"critic_loss": loss.item()}

    def _update_actor(
        self,
        agent: DDPG,
        agent_index: int,
        obs_tensors: dict[str, torch.Tensor],
        global_states: torch.Tensor,
        actions_tensor: torch.Tensor,  # (B, N, act_dim)
    ):
        """
        Paper-faithful MADDPG actor update:
        - For j ≠ agent_index: use replay-buffer actions
        - For j == agent_index: use current actor output
        """

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
        pred_action_i = agent.actor_net(obs_i)  # differentiable

        actions_all[:, agent_index, :] = pred_action_i  # keep others from buffer

        # ---------------------------------------------------------
        # Step 3: Apply M3DDPG adversarial perturbation (if enabled)
        # ---------------------------------------------------------
        if self.alpha != 0.0:
            # compute perturbation on ALL actions (but this returns detached)
            actions_adv = self._compute_adversarial_actions(
                agent_index=agent_index,
                actions=actions_all,
                global_states=global_states,
                critic=agent.critic_net,
            )

            # reinsert differentiable action for agent i
            actions_adv[:, agent_index, :] = pred_action_i
            actions_all = actions_adv

        # ---------------------------------------------------------
        # Step 4: Compute actor loss: -Q_i(x, a_1,...,a_i,...,a_N)
        # ---------------------------------------------------------
        joint_actions_flat = actions_all.reshape(batch_size, -1)
        q_val = agent.critic_net(global_states, joint_actions_flat)

        # regularization as in TF code
        reg = (pred_action_i**2).mean() * 1e-3

        actor_loss = -q_val.mean() + reg

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

        return {"actor_loss": actor_loss.item()}

    def train_policy(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        self.learn_counter += 1

        info: dict[str, Any] = {}

        for agent_index, current_agent in enumerate(self.agent_networks):
            # ---------------------------------------------------------
            # Update each agent
            # ---------------------------------------------------------
            (
                observation_tensor,
                actions_tensor,
                rewards_tensor,
                next_observation_tensor,
                dones_tensor,
                _,
                indices,
            ) = memory_sampler.sample(
                memory=memory_buffer,
                batch_size=self.batch_size,
                device=self.device,
                use_per_buffer=0,
            )

            sample_size = len(indices)

            states_tensors = observation_tensor.global_state_tensor
            next_states_tensors = next_observation_tensor.global_state_tensor

            agent_states_tensors = observation_tensor.agent_states_tensor
            next_agent_states_tensors = next_observation_tensor.agent_states_tensor

            agent_ids = list(agent_states_tensors.keys())

            # ---------------------------------------------------------
            # Build next_actions_tensor using TARGET actors
            # ---------------------------------------------------------
            next_actions = []
            for agent, agent_id in zip(self.agent_networks, agent_ids):
                obs_next_j = next_agent_states_tensors[agent_id]
                next_action_j = agent.target_actor_net(obs_next_j)
                next_actions.append(next_action_j)

            next_actions_tensor = torch.stack(next_actions, dim=1)

            print(next_actions_tensor.shape, next_actions_tensor.mean())

            # Flatten replay-buffer actions for this batch
            joint_actions = actions_tensor.reshape(sample_size, -1)

            print(joint_actions.shape, joint_actions.mean())

            # ---------------------------------------------------------
            # Critic update for this agent
            # ---------------------------------------------------------
            rewards_i = rewards_tensor[:, agent_index]
            dones_i = dones_tensor[:, agent_index]

            print(rewards_i.shape, rewards_i.mean())
            print(dones_i.shape)

            critic_info = self._update_critic(
                agent=current_agent,
                agent_index=agent_index,
                global_states=states_tensors,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_states_tensors,
                next_actions_tensor=next_actions_tensor,
                dones_i=dones_i,
            )

            # ---------------------------------------------------------
            # Actor update
            # ---------------------------------------------------------
            actor_info = self._update_actor(
                agent=current_agent,
                agent_index=agent_index,
                obs_tensors=agent_states_tensors,
                global_states=states_tensors,
                actions_tensor=actions_tensor,
            )

            info[f"critic_loss_agent_{agent_index}"] = critic_info["critic_loss"]
            info[f"actor_loss_agent_{agent_index}"] = actor_info["actor_loss"]

            current_agent.update_target_networks()

        if self.learn_counter == 1:
            exit()

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
