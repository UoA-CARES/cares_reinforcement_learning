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

import cares_reinforcement_learning.memory.memory_sampler as memory_sampler
import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import Algorithm
from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer
from cares_reinforcement_learning.networks.DDPG import Actor, Critic
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import SARLObservation
from cares_reinforcement_learning.util.configurations import DDPGConfig


class DDPG(Algorithm[SARLObservation]):
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
        self, observation: SARLObservation, evaluation: bool = False
    ) -> np.ndarray:
        # pylint: disable-next=unused-argument
        state = observation.vector_state

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
        actions_pred = self.actor_net(states)
        actor_q = self.critic_net(states, actions_pred)
        self.critic_net.train()

        actor_loss = -actor_q.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info = {"actor_loss": actor_loss.item()}
        return info

    def train_policy(
        self,
        memory_buffer: MemoryBuffer[SARLObservation],
        training_context: EpisodeContext,
    ) -> dict[str, Any]:

        # Use the helper to sample and prepare tensors in one step
        (
            observation_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor,
            dones_tensor,
            _,
            _,
        ) = memory_sampler.sample(
            memory=memory_buffer,
            batch_size=self.batch_size,
            device=self.device,
            use_per_buffer=0,  # DDPG uses uniform sampling
        )

        info: dict[str, Any] = {}

        # Update Critic
        critic_info = self._update_critic(
            observation_tensor.vector_state_tensor,
            actions_tensor,
            rewards_tensor,
            next_observation_tensor.vector_state_tensor,
            dones_tensor,
        )
        info |= critic_info

        # Update Actor
        actor_info = self._update_actor(observation_tensor.vector_state_tensor)
        info |= actor_info

        self.update_target_networks()

        return info

    def update_target_networks(self) -> None:
        hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
        hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

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
