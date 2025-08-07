"""
Original Paper: https://arxiv.org/abs/1509.06461

code based on: https://github.com/dxyang/DQN_pytorch
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.memory import ManageBuffers


class DoubleDQN:
    def __init__(
        self,
        network: torch.nn.Module,
        gamma: float,
        tau: float,
        network_lr: float,
        device: torch.device,
    ):
        self.type = "value"
        self.device = device

        self.network = network.to(self.device)
        self.target_network = copy.deepcopy(self.network).to(self.device)

        self.gamma = gamma
        self.tau = tau

        self.network_optimiser = torch.optim.Adam(
            self.network.parameters(), lr=network_lr
        )

    def select_action_from_policy(self, state: np.ndarray) -> int:
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values).item()
        self.network.train()
        return action

    def train_policy(self, memory:ManageBuffers, batch_size: int) -> None:
        experiences = memory.short_term_memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        q_values = self.network(states)
        next_q_values = self.network(next_states)
        next_q_state_values = self.target_network(next_states)

        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(
            1, torch.max(next_q_values, 1)[1].unsqueeze(1)
        ).squeeze(1)

        q_target = rewards + self.gamma * (1 - dones) * next_q_value

        loss = F.mse_loss(q_value, q_target)

        self.network_optimiser.zero_grad()
        loss.backward()
        self.network_optimiser.step()

        for target_param, param in zip(
            self.target_network.parameters(), self.network.parameters()
        ):
            target_param.data.copy_(
                param.data * self.tau + target_param.data * (1.0 - self.tau)
            )

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.network.state_dict(), f"{path}/{filename}_network.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.network.load_state_dict(torch.load(f"{path}/{filename}_network.pht"))
        logging.info("models has been loaded...")
