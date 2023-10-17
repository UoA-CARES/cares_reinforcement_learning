
import torch.nn as nn
import torch.nn.functional as F

class Critic(nn.Module):
    def __init__(self, observation_size):
        super(Critic, self).__init__()

        self.hidden_size = [1024, 1024]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(observation_size, self.hidden_size[0])
        self.h_linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_3 = nn.Linear(self.hidden_size[1], 1)


    def forward(self, state):
        q1 = F.relu(self.h_linear_1(state))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = self.h_linear_3(q1)
        return q1
