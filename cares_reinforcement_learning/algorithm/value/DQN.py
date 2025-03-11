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
from cares_reinforcement_learning.networks.DQN import Network as DQNNetwork
from cares_reinforcement_learning.networks.NoisyNet import Network as NoisyNetwork
from cares_reinforcement_learning.networks.DuelingDQN import (
    Network as DuelingDQNNetwork,
)
from cares_reinforcement_learning.util.configurations import DQNConfig


class DQN:
    def __init__(
        self,
        network: DQNNetwork | DuelingDQNNetwork | NoisyNetwork,
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

        self.use_double_dqn = config.use_double_dqn

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
        best_next_q_values = torch.max(next_q_values_target, dim=1).values

        q_target = rewards_tensor + self.gamma * (1 - dones_tensor) * best_next_q_values

        # Update the Network
        loss = F.mse_loss(best_q_values, q_target)
        return loss

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

        loss = F.mse_loss(best_q_values, q_target)
        return loss

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        info = {}

        # Generate Q Values given state at time t and t + 1
        if self.use_double_dqn:
            loss = self._double_dqn_loss(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
            )
        else:
            loss = self._dqn_loss(
                states_tensor,
                actions_tensor,
                rewards_tensor,
                next_states_tensor,
                dones_tensor,
            )

        info["loss"] = loss.item()

        self.network_optimiser.zero_grad()
        loss.backward()

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), max_norm=self.max_grad_norm
            )

        self.network_optimiser.step()

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
