import torch
import torch.nn.functional as F
from torch import nn


# class Critic(nn.Module):
#     def __init__(self, observation_size: int):
#         super().__init__()
#         hidden_sizes = [1024, 1024]

#         self.net = nn.Sequential(
#             nn.Linear(observation_size, hidden_sizes[0]),
#             nn.Tanh(),
#             nn.Linear(hidden_sizes[0], hidden_sizes[1]),
#             nn.Tanh(),
#             nn.Linear(hidden_sizes[1], 1),
#         )

#     def forward(self, state: torch.Tensor) -> torch.Tensor:
#         return self.net(state)

class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        # Shared layers
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 1024),
            nn.Tanh(),
            # nn.ReLU(),
            nn.Linear(1024,1024),
            nn.Tanh()
            # nn.ReLU()
        )
        
        # Output layer for state-value function
        self.value_fc = nn.Linear(1024, 1)

    def forward(self, state):
        x = self.fc(state)
        value = self.value_fc(x)  # Output the state-value estimate
        return value
