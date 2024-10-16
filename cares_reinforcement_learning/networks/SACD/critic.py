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
            hidden_size = [256, 256]

        self.hidden_size = hidden_size
        self.num_actions = num_actions

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.num_actions),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.num_actions),
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2
