"""
Code based on: 
https://github.com/Kchu/DeepRL_PyTorch/blob/master/Distributional_RL/1_C51.py 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Network(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate, num_atoms = 51):
        super(Network, self).__init__()

        self.hidden_size = [1024, 1024]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=num_actions * num_atoms)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

        self.num_actions = num_actions
        self.num_atoms = num_atoms

    def forward(self, state):
        x = F.relu(self.h_linear_1(state))
        x = F.relu(self.h_linear_2(x))
       
        # probability distribution on the 3rd dimension 
        x = F.softmax(self.h_linear_3(x).view(state.size(0), self.num_actions, self.num_atoms), dim = 2)
        return x
