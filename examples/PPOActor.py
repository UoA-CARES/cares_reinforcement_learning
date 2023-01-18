import torch
import torch.nn as nn


class PolicyNetworkSAC(nn.Module):
    def __init__(self, vector_size, num_actions):
        super(PolicyNetworkSAC, self).__init__()

        self.log_std_min = -20
        self.log_std_max = 2

        self.num_actions = num_actions
        self.input_size  = vector_size
        self.hidden_size = [128, 64, 32]

        self.linear1 = nn.Linear(self.input_size,     self.hidden_size[0])
        self.linear2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])

        self.mean_linear    = nn.Linear(self.hidden_size[1], self.num_actions)
        self.log_std_linear = nn.Linear(self.hidden_size[1], self.num_actions)

    def forward(self, state):
        x = torch.relu(self.linear1(state))
        x = torch.relu(self.linear2(x))

        mean    = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)

        return mean, log_std
