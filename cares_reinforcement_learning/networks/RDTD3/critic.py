import torch
import torch.nn as nn
import torch.nn.functional as F


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int):
        super(Critic, self).__init__()

        self.hidden_size = [256, 256]  # [256, 256], [1024, 1024]

        # Q1 architecture
        self.h_linear_1 = nn.Linear(observation_size + num_actions, self.hidden_size[0])
        self.h_linear_2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_3 = nn.Linear(self.hidden_size[1], 1 + 1 + observation_size)

        # Q2 architecture
        self.h_linear_12 = nn.Linear(
            observation_size + num_actions, self.hidden_size[0]
        )
        self.h_linear_22 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.h_linear_32 = nn.Linear(self.hidden_size[1], 1 + 1 + observation_size)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)

        output_1 = F.relu(self.h_linear_1(obs_action))
        output_1 = F.relu(self.h_linear_2(output_1))
        output_1 = self.h_linear_3(output_1)

        output_2 = F.relu(self.h_linear_12(obs_action))
        output_2 = F.relu(self.h_linear_22(output_2))
        output_2 = self.h_linear_32(output_2)

        return output_1, output_2
