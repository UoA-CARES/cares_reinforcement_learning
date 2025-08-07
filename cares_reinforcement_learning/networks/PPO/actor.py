# import torch
# import torch.nn as nn
# from torch.distributions import Normal


# class Actor(nn.Module):
#     def __init__(self, observation_size: int, num_actions: int):
#         super().__init__()
#         hidden_sizes = [1024, 1024]

#         self.net = nn.Sequential(
#             nn.Linear(observation_size, hidden_sizes[0]),
#             nn.Tanh(),
#             nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#             nn.Tanh(),
#             nn.Linear(hidden_sizes[1], num_actions),
#         )

#         # Learnable log_std
#         self.log_std = nn.Parameter(torch.zeros(num_actions))

#     def forward(self, state: torch.Tensor):
#         mean = self.net(state)
#         action_logstd = self.log_std.expand_as(mean)
#         action_std = torch.exp(action_logstd)
#         return mean, action_std

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        # Shared layers
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(1024,1024),
            nn.Tanh()
            # nn.ReLU(),
        )
        
        # Output layer for the mean of action distribution
        self.actor_fc = nn.Linear(1024, action_dim)
        
        # For continuous actions, we need a fixed standard deviation (you could also learn this)
        self.log_std = nn.Parameter(torch.zeros(1, action_dim))  # Log of standard deviation

    def forward(self, state):
        x = self.fc(state)
        mean = self.actor_fc(x)
        std = torch.exp(self.log_std)  # The standard deviation is exp of log_std for stability
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        dist = Normal(mean, std)
        action = dist.sample()  # Sample from the action distribution
        log_prob = dist.log_prob(action).sum(dim=-1)  # Log probability of the action
        return action, log_prob
