import torch
from torch import nn


class DistributedCritic(nn.Module):
    def __init__(
        self,
        observation_size: int,
        action_num: int,
        hidden_size: list[int] | None = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]

        self.hidden_size = hidden_size

        self.mean_layer = nn.Sequential(
            nn.Linear(observation_size + action_num, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        self.std_layer = nn.Sequential(
            nn.Linear(observation_size + action_num, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
            nn.Softplus(),
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        u = self.mean_layer(obs_action)
        std = self.std_layer(obs_action) + 1e-6
        return u, std
