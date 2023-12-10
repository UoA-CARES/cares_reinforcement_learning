import torch
from torch import nn


class Critic(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()

        self.hidden_size = [256, 256]

        # Q1 architecture
        self.q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # Q2 architecture
        self.q2 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def forward(self, state, action):
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.q1(obs_action)
        q2 = self.q2(obs_action)
        return q1, q2
