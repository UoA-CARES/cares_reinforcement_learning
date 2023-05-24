"""
code based on: https://github.com/dxyang/DQN_pytorch
"""

import os
import copy
import torch
import logging
import numpy as np
import torch.nn.functional as F


class DoubleDQN:

    def __init__(self,
                 network,
                 gamma,
                 tau,
                 device):

        self.type = "value"
        self.network = network.to(device)
        self.target_network = copy.deepcopy(self.network).to(device)

        self.gamma  = gamma
        self.tau    = tau
        self.device = device

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            q_values = self.network(state_tensor)
            action   = torch.argmax(q_values).item()
        return action

    def train_policy(self, experiences):
        states, actions, rewards, next_states, dones = experiences
        info = {}

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        q_values = self.network(states)
        next_q_values = self.network(next_states)
        next_q_state_values = self.target_network(next_states)

        info["q_values_min"] = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_state_values.gather(1, torch.max(next_q_values, 1)[1].unsqueeze(1)).squeeze(1)

        info["q_target"] = rewards + self.gamma * (1 - dones) * next_q_value

        loss = F.mse_loss(info["q_values_min"], info["q_target"])
        self.network.optimiser.zero_grad()
        loss.backward()
        self.network.optimiser.step()

        for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        return info

    def save_models(self,filename, filepath='models'):
        path = f"{filepath}/models" if filepath is not 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.network.state_dict(),  f'{path}/{filename}_network.pht')
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath is not 'models' else filepath
        
        self.network.load_state_dict(torch.load(f'{path}/{filename}_network.pht'))
        logging.info("models has been loaded...")
