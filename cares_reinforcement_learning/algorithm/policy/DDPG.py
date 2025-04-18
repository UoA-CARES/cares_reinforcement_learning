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
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.DDPG import Actor, Critic
from cares_reinforcement_learning.util.configurations import DDPGConfig


class DDPG(Algorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: DDPGConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", device=device)

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
        state: np.ndarray,
        evaluation: bool = False,
    ) -> np.ndarray:
        # pylint: disable-next=unused-argument

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

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        experiences = memory.sample_uniform(batch_size)
        (states, actions, rewards, next_states, dones, _) = experiences

        batch_size = len(states)

        # Convert into tensor
        states_tensors = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensors = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensors = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensors = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensors = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards_tensors = rewards_tensors.reshape(batch_size, 1)
        dones_tensors = dones_tensors.reshape(batch_size, 1)

        info: dict[str, Any] = {}

        # Update Critic
        critic_info = self._update_critic(
            states_tensors,
            actions_tensors,
            rewards_tensors,
            next_states_tensors,
            dones_tensors,
        )
        info |= critic_info

        # Update Actor
        actor_info = self._update_actor(states_tensors)
        info |= actor_info

        hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
