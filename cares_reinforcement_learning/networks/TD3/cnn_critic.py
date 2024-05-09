import torch
from torch import nn

from cares_reinforcement_learning.util.common import NatureCNN


class CNNCritic(nn.Module):
    def __init__(self, observation_size: tuple[int], num_actions: int):
        super().__init__()

        self.hidden_size = [256, 256]

        self.nature_cnn_one = NatureCNN(observation_size=observation_size)
        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(512 + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        self.nature_cnn_two = NatureCNN(observation_size=observation_size)

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(512 + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:

        obs_one = self.nature_cnn_one(state)
        obs_action_one = torch.cat([obs_one, action], dim=1)
        q1 = self.Q1(obs_action_one)

        obs_two = self.nature_cnn_two(state)
        obs_action_two = torch.cat([obs_two, action], dim=1)
        q2 = self.Q2(obs_action_two)

        return q1, q2
