"""
Original Paper: https://arxiv.org/pdf/1509.02971v5.pdf
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
from cares_reinforcement_learning.util.configurations import MADDPGConfig
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class MADDPG(Algorithm):
    def __init__(
        self,
        agents: list[DDPG],
        config: MADDPGConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agents = agents
        self.num_agents = len(agents)

        self.gamma = config.gamma
        self.tau = config.tau

    def select_action_from_policy(
        self,
        action_context: ActionContext,
    ):
        state = action_context.state

        assert isinstance(state, dict)

        actions = []
        for i, agent in enumerate(self.agents):
            obs_i = state["obs"][i]
            agent_action_context = ActionContext(
                state=obs_i,
                evaluation=action_context.evaluation,
                available_actions=action_context.available_actions[i],
            )
            action = agent.select_action_from_policy(agent_action_context)
            actions.append(action)
        return actions

    def _update_critic(
        self,
        agent: DDPG,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,
        rewards_i: torch.Tensor,
        next_global_states: torch.Tensor,
        next_joint_actions: torch.Tensor,
        dones_i: torch.Tensor,
    ):
        with torch.no_grad():
            target_q = agent.target_critic_net(next_global_states, next_joint_actions)
            q_target = rewards_i + self.gamma * (1 - dones_i) * target_q

        q_values = agent.critic_net(global_states, joint_actions)

        loss = F.mse_loss(q_values, q_target)

        agent.critic_net_optimiser.zero_grad()
        loss.backward()
        agent.critic_net_optimiser.step()

        return {"critic_loss": loss.item()}

    def _update_actor(
        self,
        agent: DDPG,
        agent_index: int,
        obs_i: torch.Tensor,  # (batch, obs_dim)
        actions_all: torch.Tensor,  # (batch, n_agents, act_dim)
        global_states: torch.Tensor,  # (batch, obs_dim*n_agents)
    ):
        batch_size = obs_i.shape[0]

        # -------------------------
        # 1. Predict agent i's action
        # -------------------------
        pred_action_i = agent.actor_net(obs_i)  # (batch, act_dim)

        # -------------------------
        # 2. Build pred joint actions WITHOUT loop
        # -------------------------
        pred_actions_tensor = actions_all.clone()  # (batch, n_agents, act_dim)
        pred_actions_tensor[:, agent_index, :] = pred_action_i

        # Flatten to joint action vector
        joint_actions_pred = pred_actions_tensor.reshape(batch_size, -1)
        # shape: (batch, n_agents * act_dim)

        # -------------------------
        # 3. Actor loss
        # -------------------------
        q_val = agent.critic_net(global_states, joint_actions_pred)
        actor_loss = -q_val.mean()

        # -------------------------
        # 4. Apply gradients
        # -------------------------
        agent.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        agent.actor_net_optimiser.step()

        return {"actor_loss": actor_loss.item()}

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        memory = training_context.memory
        batch_size = training_context.batch_size

        # Use training_utils to sample and prepare batch
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            _,
            _,
        ) = tu.sample_marl_batch_to_tensors(
            memory,
            batch_size,
            self.device,
            use_per_buffer=0,
        )

        # Unpack useful tensors
        obs_n = states_tensor["obs"]
        next_obs_n = next_states_tensor["obs"]
        global_state = states_tensor["state"]
        next_global_state = next_states_tensor["state"]

        info: dict[str, Any] = {}

        next_actions_tensor = torch.stack(
            [
                agent.target_actor_net(next_obs_n[:, i, :])
                for i, agent in enumerate(self.agents)
            ],
            dim=1,
        )
        next_joint_actions = next_actions_tensor.reshape(batch_size, -1)

        next_joint_actions = next_actions_tensor.reshape(batch_size, -1)

        joint_actions = actions_tensor.reshape(batch_size, -1)

        for i, agent in enumerate(self.agents):
            rewards_i = rewards_tensor[:, i].unsqueeze(-1)
            dones_i = dones_tensor[:, i].unsqueeze(-1)

            critic_info = self._update_critic(
                agent=agent,
                global_states=global_state,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_global_state,
                next_joint_actions=next_joint_actions,
                dones_i=dones_i,
            )

            obs_i = obs_n[:, i, :]

            actor_info = self._update_actor(
                agent=agent,
                obs_i=obs_i,
                actions_all=actions_tensor,
                global_states=global_state,
                agent_index=i,
            )

            info[f"critic_loss_agent_{i}"] = critic_info["critic_loss"]
            info[f"actor_loss_agent_{i}"] = actor_info["actor_loss"]

        for agent in self.agents:
            agent.update_target_networks()

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, agent in enumerate(self.agents):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for i, agent in enumerate(self.agents):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
