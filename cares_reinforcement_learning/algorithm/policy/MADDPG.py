"""
Original Paper: https://arxiv.org/pdf/1509.02971v5.pdf
"""

import logging
import os
from typing import Any

import torch
import torch.nn.functional as F

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

        self.max_grad_norm = config.max_grad_norm

        self.alpha = config.alpha  # adversarial perturbation scale

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

        # --- Step 3: critic regression on *current* joint_actions (unperturbed) ---
        q_values = agent.critic_net(global_states, joint_actions)
        loss = F.mse_loss(q_values, q_target)

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
        obs_i: torch.Tensor,
        actions_all: torch.Tensor,  # (B, N, act_dim)
        global_states: torch.Tensor,
    ):
        batch_size = obs_i.shape[0]

        # --- Step 1: current policy action for agent i ---
        pred_action_i = agent.actor_net(obs_i)

        # --- Step 2: insert predicted action for agent i ---
        pred_actions_tensor = actions_all.clone()
        pred_actions_tensor[:, agent_index, :] = pred_action_i

        # --- Step 3: adversarially perturb *other* agents ---
        if self.alpha != 0.0:
            actions_adv = self._compute_adversarial_actions(
                agent_index=agent_index,
                actions=pred_actions_tensor,
                global_states=global_states,
                critic=agent.critic_net,  # online critic
            )
            # Reinsert pred_action_i so gradients flow correctly
            pred_actions_tensor = actions_adv.clone()
            pred_actions_tensor[:, agent_index, :] = pred_action_i

        # --- Step 4: actor loss ---
        joint_actions_pred = pred_actions_tensor.view(batch_size, -1)
        q_val = agent.critic_net(global_states, joint_actions_pred)
        actor_loss = -q_val.mean()

        agent.actor_net_optimiser.zero_grad()
        actor_loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                agent.actor_net.parameters(), self.max_grad_norm
            )

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
        )  # shape (batch, n_agents, act_dim)

        joint_actions = actions_tensor.reshape(batch_size, -1)

        for i, agent in enumerate(self.agents):
            rewards_i = rewards_tensor[:, i].unsqueeze(-1)
            dones_i = dones_tensor[:, i].unsqueeze(-1)

            critic_info = self._update_critic(
                agent=agent,
                agent_index=i,
                global_states=global_state,
                joint_actions=joint_actions,
                rewards_i=rewards_i,
                next_global_states=next_global_state,
                next_actions_tensor=next_actions_tensor,
                dones_i=dones_i,
            )

            obs_i = obs_n[:, i, :]

            actor_info = self._update_actor(
                agent=agent,
                agent_index=i,
                obs_i=obs_i,
                actions_all=actions_tensor,
                global_states=global_state,
            )

            info[f"critic_loss_agent_{i}"] = critic_info["critic_loss"]
            info[f"actor_loss_agent_{i}"] = actor_info["actor_loss"]

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
