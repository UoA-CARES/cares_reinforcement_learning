"""
Original Paper: https://arxiv.org/abs/1312.5602
"""

import logging
import os

import numpy as np
import torch
import torch.nn.functional as F


from cares_reinforcement_learning.memory import MemoryBuffer


class DQN:
    def __init__(
        self,
        network: torch.nn.Module,
        gamma: float,
        network_lr: float,
        device: torch.device,
    ):
        self.type = "value"
        self.network = network.to(device)
        self.device = device
        self.gamma = gamma

        self.network_optimiser = torch.optim.Adam(
            self.network.parameters(), lr=network_lr
        )

    def select_action_from_policy(self, state):
        self.network.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network(state_tensor)
            action = torch.argmax(q_values).item()
        self.network.train()
        return action

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Generate Q Values given state at time t and t + 1
        q_values = self.network(states)
        next_q_values = self.network(next_states)

        best_q_values = q_values[torch.arange(q_values.size(0)), actions]
        best_next_q_values = torch.max(next_q_values, dim=1).values

        q_target = rewards + self.gamma * (1 - dones) * best_next_q_values

        # Update the Network
        loss = F.mse_loss(best_q_values, q_target)
        self.network_optimiser.zero_grad()
        loss.backward()
        self.network_optimiser.step()

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
