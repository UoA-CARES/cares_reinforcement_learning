import torch
from torch import nn

from cares_reinforcement_learning.util.common import NatureCNN


class CNNActor(nn.Module):
    def __init__(self, observation_size: tuple[int], num_actions: int):
        super().__init__()

        self.hidden_size = [256, 256]

        self.nature_cnn = NatureCNN(observation_size=observation_size)

        self.act_net = nn.Sequential(
            nn.Linear(512, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.nature_cnn(state)
        output = self.act_net(output)
        return output
