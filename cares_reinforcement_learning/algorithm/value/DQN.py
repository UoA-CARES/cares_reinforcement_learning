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
from cares_reinforcement_learning.networks.DuelingDQN import (
    Network as DuelingDQNNetwork,
)
from cares_reinforcement_learning.util.configurations import DQNConfig, DuelingDQNConfig


class DQN:
    def __init__(
        self,
        network: DQNNetwork | DuelingDQNNetwork,
        config: DQNConfig | DuelingDQNConfig,
        device: torch.device,
    ):
        self.type = "value"
        self.network = network.to(device)
        self.target_network = copy.deepcopy(self.network).to(device)
        self.target_network.eval()

        self.device = device
        self.gamma = config.gamma
        self.target_update_freq = config.target_update_freq

        self.network_optimiser = torch.optim.Adam(
            self.network.parameters(), lr=config.lr
        )

        self.train_step_counter = 0

    def select_action_from_policy(self, state) -> float:
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values).item()
        self.network.train()
        return action

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Generate Q Values given state at time t and t + 1
        q_values = self.network(states_tensor)
        next_q_values = self.target_network(next_states_tensor)

        best_q_values = q_values[torch.arange(q_values.size(0)), actions_tensor]
        best_next_q_values = torch.max(next_q_values, dim=1).values

        q_target = rewards_tensor + self.gamma * (1 - dones_tensor) * best_next_q_values

        # Update the Network
        loss = F.mse_loss(best_q_values, q_target)
        self.network_optimiser.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.network_optimiser.step()

        # return info
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            hlp.hard_update_params(self.network, self.target_network)
        
        return {"loss": loss.item()}


    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.network.state_dict(), f"{filepath}/{filename}_network.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.network.load_state_dict(torch.load(f"{filepath}/{filename}_network.pht"))
        logging.info("models has been loaded...")
