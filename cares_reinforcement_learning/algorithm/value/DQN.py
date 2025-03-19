"""
Original Paper: https://arxiv.org/abs/1312.5602
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.C51 import Network as C51Network
from cares_reinforcement_learning.networks.DQN import Network as DQNNetwork
from cares_reinforcement_learning.networks.DuelingDQN import (
    Network as DuelingDQNNetwork,
)
from cares_reinforcement_learning.networks.NoisyNet import Network as NoisyNetwork
from cares_reinforcement_learning.util.configurations import DQNConfig


class DQN:
    def __init__(
        self,
        network: DQNNetwork | DuelingDQNNetwork | NoisyNetwork | C51Network,
        config: DQNConfig,
        device: torch.device,
    ):
        self.type = "value"
        self.device = device

        self.network = network.to(device)
        self.target_network = copy.deepcopy(self.network).to(device)
        self.target_network.eval()

        self.tau = config.tau
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq

        # Double DQN
        self.use_double_dqn = config.use_double_dqn

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.min_priority = config.min_priority
        self.per_alpha = config.per_alpha

        # C51
        self.use_c51 = config.use_c51
        self.num_atoms = config.num_atoms
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms).to(device)

        self.max_grad_norm = config.max_grad_norm

        self.network_optimiser = torch.optim.Adam(
            self.network.parameters(), lr=config.lr
        )

        self.learn_counter = 0

    def select_action_from_policy(self, state) -> float:
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values).item()
        self.network.train()
        return action

    def _dqn_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.network(states_tensor)
        next_q_values_target = self.target_network(next_states_tensor)

        # Get Q-values for chosen actions
        best_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        # Get the best Q-values from the online network
        best_next_q_values = torch.max(next_q_values_target, dim=1).values

        q_target = rewards_tensor + self.gamma * (1 - dones_tensor) * best_next_q_values

        elementwise_loss = F.mse_loss(best_q_values, q_target, reduction="none")

        return elementwise_loss

    def _double_dqn_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
    ) -> torch.Tensor:
        q_values = self.network(states_tensor)
        next_q_values_target = self.target_network(next_states_tensor)

        # Online Used for action selection
        next_q_values = self.network(next_states_tensor)

        # Get Q-values for chosen actions
        best_q_values = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # Double DQN: Select best action from online Q-values, evaluate with target Q-values
        # Online network selects actions
        next_actions = next_q_values.argmax(dim=1, keepdim=True)
        # Target network estimates value
        best_next_q_values = next_q_values_target.gather(1, next_actions).squeeze(1)

        q_target = rewards_tensor + self.gamma * (1 - dones_tensor) * best_next_q_values

        elementwise_loss = F.mse_loss(best_q_values, q_target, reduction="none")

        return elementwise_loss

    def _c51_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)
        with torch.no_grad():
            # DQN
            next_action = self.target_network(next_states_tensor).argmax(1)
            next_dist = self.target_network.dist(next_states_tensor)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = (
                rewards_tensor.unsqueeze(1)
                + (1 - dones_tensor.unsqueeze(1)) * self.gamma * self.support
            )

            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.network.dist(states_tensor)
        log_p = torch.log(dist[range(batch_size), actions_tensor])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _double_c51_loss(
        self,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.network(next_states_tensor).argmax(1)
            next_dist = self.target_network.dist(next_states_tensor)
            next_dist = next_dist[range(batch_size), next_action]

            t_z = (
                rewards_tensor.unsqueeze(1)
                + (1 - dones_tensor.unsqueeze(1)) * self.gamma * self.support
            )
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size)
                .long()
                .unsqueeze(1)
                .expand(batch_size, self.num_atoms)
                .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.network.dist(states_tensor)
        log_p = torch.log(dist[range(batch_size), actions_tensor])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.learn_counter += 1

        if self.use_per_buffer:
            experiences = memory.sample_priority(batch_size)
            states, actions, rewards, next_states, dones, indices, weights = experiences
            weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)
        else:
            experiences = memory.sample_uniform(batch_size)
            states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        if self.use_per_buffer:
            weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)

        info = {}

        if self.use_c51:
            compute_c51_loss_fn = (
                self._double_c51_loss if self.use_double_dqn else self._c51_loss
            )
            elementwise_loss = compute_c51_loss_fn(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
                batch_size,
            )
        else:
            compute_dqn_loss_fn = (
                self._double_dqn_loss if self.use_double_dqn else self._dqn_loss
            )
            elementwise_loss = compute_dqn_loss_fn(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
            )

        if self.use_per_buffer:
            # Update the Priorities
            priorities = (
                elementwise_loss.clamp(self.min_priority)
                .pow(self.per_alpha)
                .cpu()
                .data.numpy()
                .flatten()
            )

            memory.update_priorities(indices, priorities)

            loss = torch.mean(elementwise_loss * weights_tensor)
        else:
            # Calculate loss
            loss = elementwise_loss.mean()

        info["loss"] = loss.item()

        self.network_optimiser.zero_grad()
        loss.backward()

        # Apply gradient clipping if max_grad_norm is set
        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=self.max_grad_norm
            )

        self.network_optimiser.step()

        # Update target network - a tau of 1.0 equates to a hard update.
        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.network, self.target_network, self.tau)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.network.state_dict(), f"{filepath}/{filename}_network.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.network.load_state_dict(torch.load(f"{filepath}/{filename}_network.pht"))
        logging.info("models has been loaded...")
