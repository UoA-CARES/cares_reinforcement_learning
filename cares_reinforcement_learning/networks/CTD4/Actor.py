
from torch import nn


class Actor(nn.Module):
    def __init__(self, observation_size, action_num):
        super().__init__()

        self.hidden_size = [256, 256]

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], action_num),
            nn.Tanh()
        )

    def forward(self, state):
        output   = self.act_net(state)
        return output
