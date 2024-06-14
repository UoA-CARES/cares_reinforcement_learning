import torch
from torch import nn
from torch.nn import functional as F


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()

        self.hidden_size = [256, 256]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(observation_size + num_actions, self.hidden_size[0])
        self.h_linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_3 = nn.Linear(self.hidden_size[1], 1)

        # Q2 architecture
        self.h_linear_12 = nn.Linear(
            observation_size + num_actions, self.hidden_size[0]
        )
        self.h_linear_22 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_32 = nn.Linear(self.hidden_size[1], 1)
        
        self.apply(self.weights_init_)
        

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)

        q1 = F.relu(self.h_linear_1(obs_action))
        q1 = F.relu(self.h_linear_2(q1))
        q1 = self.h_linear_3(q1)

        q2 = F.relu(self.h_linear_12(obs_action))
        q2 = F.relu(self.h_linear_22(q2))
        q2 = self.h_linear_32(q2)

        return q1, q2
    
    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)
