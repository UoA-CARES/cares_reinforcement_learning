
import torch.nn as nn
import torch.optim as optim


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions, learning_rate):
        super(Actor, self).__init__()

        self.hidden_size = [1024, 1024]

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh()
        )

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        output   = self.act_net(state)
        return output
