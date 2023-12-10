import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions):
        super(Actor, self).__init__()

        self.hidden_size = [1024, 1024]
        self.log_sig_min = -20
        self.log_sig_max = 2

        self.h_linear_1 = nn.Linear(
            in_features=observation_size, out_features=self.hidden_size[0]
        )
        self.h_linear_2 = nn.Linear(
            in_features=self.hidden_size[0], out_features=self.hidden_size[1]
        )

        self.mean_linear = nn.Linear(
            in_features=self.hidden_size[1], out_features=num_actions
        )
        self.log_std_linear = nn.Linear(
            in_features=self.hidden_size[1], out_features=num_actions
        )

    def forward(self, state):
        x = F.relu(self.h_linear_1(state))
        x = F.relu(self.h_linear_2(x))

        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)

        x_t = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t

        epsilon = 1e-6
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean)

        return action, log_prob, mean
