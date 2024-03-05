import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Critic, self).__init__()

        self.hidden_size = [1024, 1024]  # [256, 256], [1024, 1024]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(observation_size + num_actions, self.hidden_size[0])
        self.h_linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_3 = nn.Linear(self.hidden_size[1], 1 + 1 + observation_size)

        # Q2 architecture
        self.h_linear_12 = nn.Linear(observation_size + num_actions, self.hidden_size[0])
        self.h_linear_22 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_32 = nn.Linear(self.hidden_size[1],  1 + 1 + observation_size)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)  # AdamW, Adam

    def forward(self, state, action):
        obs_action = torch.cat([state, action], dim=1)

        q1 = F.relu(self.h_linear_1(obs_action))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = self.h_linear_3(q1)

        q2 = F.relu(self.h_linear_12(obs_action))
        q2 = F.relu(self.h_linear_22(q2))
        q2 = self.h_linear_32(q2)

        return q1, q2

    def Q1(self, state, action):
        obs_action = torch.cat([state, action], 1)

        q1 = F.relu(self.h_linear_1(obs_action))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = self.h_linear_3(q1)
        return q1
