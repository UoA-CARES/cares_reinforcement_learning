import torch
import torch.nn.functional as F
from torch import nn


class Critic(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()

        self.hidden_size = [1024, 1024]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(observation_size + num_actions, self.hidden_size[0])
        self.h_linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_3 = nn.Linear(self.hidden_size[1], 1)

    def forward(self, state, action):
        obs_action = torch.cat([state, action], dim=1)

        q1 = F.relu(self.h_linear_1(obs_action))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = self.h_linear_3(q1)

        return q1
