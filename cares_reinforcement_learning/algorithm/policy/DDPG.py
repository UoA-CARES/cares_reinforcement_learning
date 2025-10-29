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
from cares_reinforcement_learning.networks.DDPG import Actor, Critic
from cares_reinforcement_learning.util.configurations import DDPGConfig
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class DDPG(Algorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: DDPGConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = config.gamma
        self.tau = config.tau

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
        )

    def select_action_from_policy(
        self,
        action_context: ActionContext,
    ) -> np.ndarray:
        # pylint: disable-next=unused-argument
        state = action_context.state

        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> dict[str, Any]:
        with torch.no_grad():
            self.target_actor_net.eval()
            next_actions = self.target_actor_net(next_states)
            self.target_actor_net.train()

            target_q_values = self.target_critic_net(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values = self.critic_net(states, actions)

        critic_loss = F.mse_loss(q_values, q_target)
        self.critic_net_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net_optimiser.step()

        info = {
            "critic_loss": critic_loss.item(),
        }

        return info

    def _update_actor(self, states: torch.Tensor) -> dict[str, Any]:
        self.critic_net.eval()
        actor_q = self.critic_net(states, self.actor_net(states))
        self.critic_net.train()

        actor_loss = -actor_q.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info = {"actor_loss": actor_loss.item()}
        return info

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        memory = training_context.memory
        batch_size = training_context.batch_size

        # Use the helper to sample and prepare tensors in one step
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            _,
            _,
        ) = tu.sample_batch_to_tensors(
            memory=memory,
            batch_size=batch_size,
            device=self.device,
            use_per_buffer=0,  # DDPG uses uniform sampling
        )

        info: dict[str, Any] = {}

        # Update Critic
        critic_info = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info |= critic_info

        # Update Actor
        actor_info = self._update_actor(states_tensor)
        info |= actor_info

        hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        logging.info("models and optimisers have been loaded...")
