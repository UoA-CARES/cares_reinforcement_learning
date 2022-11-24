import torch
import torch.nn as nn
import torch.optim as optim
from gym import Space

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")


class Actor(nn.Module):
    def __init__(self, obs_space: Space, act_space: Space, learning_rate):
        super(Actor, self).__init__()

        self.input_size = obs_space.shape[0]
        self.num_actions = act_space.shape[0]

        self.max_action = act_space.high.max()
        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x)) * self.max_action
        return x
