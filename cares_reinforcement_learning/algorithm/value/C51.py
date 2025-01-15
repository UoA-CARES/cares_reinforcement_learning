"""
Original Paper: https://arxiv.org/pdf/1707.06887
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.C51 import Network
from cares_reinforcement_learning.util.configurations import C51Config


class C51:
    def __init__(
        self,
        network: Network,
        config: C51Config,
        device: torch.device,
    ):
        self.type = "value"
        self.device = device

        self.network = network.to(device)
        self.target_network = copy.deepcopy(self.network).to(self.device)

        self.gamma = config.gamma
        self.v_min = config.v_min
        self.v_max = config.v_max
        self.num_atoms = config.num_atoms

        self.network_optimiser = torch.optim.Adam(
            self.network.parameters(), lr=config.lr
        )

    def select_action_from_policy(self, state) -> float:
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values, _ = self.network(state_tensor)
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

        info = {}

        delta_z = float(self.v_max - self.v_min) / (self.num_atoms - 1)

        # Generate Q Values given state at time t and t + 1
        with torch.no_grad():
            next_q_values, next_dist = self.target_network(next_states_tensor)
            next_action = next_q_values.argmax(dim=1)
            next_dist = next_dist[range(batch_size), next_action]

            support = self.network.support
            t_z = rewards_tensor + self.gamma * (1 - dones_tensor) * support
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

        # Update the Network
        _, dist = self.network(states_tensor)
        log_p = torch.log(dist[range(batch_size), actions_tensor])

        loss = -(proj_dist * log_p).sum(1).mean()

        self.network_optimiser.zero_grad()
        loss.backward()
        self.network_optimiser.step()

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
