"""
Original Paper: https://arxiv.org/pdf/1509.02971v5.pdf
"""

import copy
import logging
import os
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer


class DDPG:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        self.gamma = gamma
        self.tau = tau

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
        self,
        state: np.ndarray,
        evaluation: Optional[bool] = False,
        noise_scale: float = 0,
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
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
    ) -> float:
        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_q_values = self.target_critic_net(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values = self.critic_net(states, actions)

        critic_loss = F.mse_loss(q_values, q_target)
        self.critic_net_optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net_optimiser.step()

        return critic_loss.item()

    def _update_actor(self, states: torch.Tensor) -> float:
        actor_q = self.critic_net(states, self.actor_net(states))
        actor_loss = -actor_q.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        return actor_loss.item()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        info = {}

        # Update Critic
        critic_loss = self._update_critic(states, actions, rewards, next_states, dones)
        info["critic_loss"] = critic_loss

        # Update Actor
        actor_loss = self._update_actor(states)
        info["actor_loss"] = actor_loss

        hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        return info

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
