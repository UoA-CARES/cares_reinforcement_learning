import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [2048, 2048]

        self.hidden_size = hidden_size

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0], bias=False),
            nn.BatchNorm1d(self.hidden_size[0], momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1], bias=False),
            nn.BatchNorm1d(self.hidden_size[1], momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(observation_size + num_actions, self.hidden_size[0], bias=False),
            nn.BatchNorm1d(self.hidden_size[0], momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1], bias=False),
            nn.BatchNorm1d(self.hidden_size[1], momentum=0.01),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2