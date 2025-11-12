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
        pass

    def _update_critic(
        self,
        agent: DDPG,
        global_states: torch.Tensor,
        joint_actions: torch.Tensor,
        rewards_i: torch.Tensor,
        next_global_states: torch.Tensor,
        next_joint_actions: torch.Tensor,
        dones: torch.Tensor,
    ):
        with torch.no_grad():
            target_q = agent.target_critic_net(next_global_states, next_joint_actions)
            q_target = rewards_i + self.gamma * (1 - dones) * target_q

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
        obs_i: torch.Tensor,
        global_states: torch.Tensor,
        joint_actions: list[torch.Tensor],
    ) -> dict[str, Any]:
        """
        Updates the actor for one agent, using its local obs and
        the centralized critic with all agents' actions.
        """
        # Recompute this agent's action to get a differentiable path
        a_i_pred = agent.actor_net(obs_i)

        # Replace only this agent's action in the joint action list
        updated_actions = joint_actions.copy()
        updated_actions[agent_index] = a_i_pred

        # Concatenate into a single joint tensor for critic input
        joint_actions_pred = torch.cat(updated_actions, dim=-1)

        # Evaluate centralized critic using the predicted joint actions
        agent.critic_net.eval()
        q_pred = agent.critic_net(global_states, joint_actions_pred)
        agent.critic_net.train()

        # Actor loss: maximize Q-value (so minimize -Q)
        actor_loss = -q_pred.mean()

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

        # Step 1. Compute next actions from target actors
        next_actions_n = [
            agent.target_actor_net(o_next)
            for agent, o_next in zip(self.agents, next_obs_n)
        ]
        next_joint_actions = torch.cat(next_actions_n, dim=-1)

        # Step 2. Compute current joint actions (from replay or current actors)
        current_actions_n = [a.detach() for a in actions_tensor]  # from replay
        joint_actions = torch.cat(current_actions_n, dim=-1)

        # Step 3. Update critics and actors per agent
        for i, agent in enumerate(self.agents):
            critic_info = self._update_critic(
                agent=agent,
                global_states=global_state,
                joint_actions=joint_actions,
                rewards_i=rewards_tensor[i],
                next_global_states=next_global_state,
                next_joint_actions=next_joint_actions,
                dones=dones_tensor,
            )

            actor_info = self._update_actor(
                agent=agent,
                agent_index=i,
                obs_i=obs_n[i],
                global_states=global_state,
                joint_actions=current_actions_n,
            )

            agent.update_target_networks()

            info[f"critic_loss_agent_{i}"] = critic_info["critic_loss"]
            info[f"actor_loss_agent_{i}"] = actor_info["actor_loss"]

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {}
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        logging.info("models and optimisers have been loaded...")
