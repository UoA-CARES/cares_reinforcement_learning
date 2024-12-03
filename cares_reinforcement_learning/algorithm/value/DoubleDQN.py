"""
Original Paper: https://arxiv.org/abs/1509.06461

code based on: https://github.com/dxyang/DQN_pytorch
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.DoubleDQN import Network
from cares_reinforcement_learning.util.configurations import DoubleDQNConfig


class DoubleDQN:
    def __init__(
        self,
        network: Network,
        config: DoubleDQNConfig,
        device: torch.device,
    ):
        self.type = "value"
        self.device = device

        self.network = network.to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)

        self.gamma = config.gamma
        self.tau = config.tau

        self.network_optimiser = torch.optim.Adam(
            self.network.parameters(), lr=config.lr
        )

    def select_action_from_policy(self, state: np.ndarray) -> float:
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

        q_values = self.network(states_tensor)
        next_q_values = self.network(next_states_tensor)
        next_q_state_values = self.target_network(next_states_tensor)

        q_value = q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(
            1, torch.max(next_q_values, 1)[1].unsqueeze(1)
        ).squeeze(1)

        q_target = rewards_tensor + self.gamma * (1 - dones_tensor) * next_q_value

        loss = F.mse_loss(q_value, q_target)

        info = {}

        self.network_optimiser.zero_grad()
        loss.backward()
        self.network_optimiser.step()

        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

        info["loss"] = loss.item()
        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.network.state_dict(), f"{filepath}/{filename}_network.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.network.load_state_dict(torch.load(f"{filepath}/{filename}_network.pht"))
        logging.info("models has been loaded...")
