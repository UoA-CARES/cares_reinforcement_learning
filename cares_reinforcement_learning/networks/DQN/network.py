import torch
import torch.nn.functional as F
from torch import nn


class Network(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [512, 512]

        self.hidden_size = hidden_size

        self.h_linear_1 = nn.Linear(
            in_features=observation_size, out_features=self.hidden_size[0]
        )
        self.h_linear_2 = nn.Linear(
            in_features=self.hidden_size[0], out_features=self.hidden_size[1]
        )
        self.h_linear_3 = nn.Linear(
            in_features=self.hidden_size[1], out_features=num_actions
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.h_linear_1(state))
        x = F.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)
        return x
