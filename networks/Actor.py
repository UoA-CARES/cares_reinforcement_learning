import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")


# class Actor(nn.Module):
#     def __init__(self, observation_size, num_actions, learning_rate):
#         super(Actor, self).__init__()
#
#         self.input_size = observation_size
#         self.num_actions = num_actions
#         self.hidden_size = [128, 64, 32]
#
#         self.h_linear_1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size[0])
#         self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
#         self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
#         self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=self.num_actions)
#
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#
#     def forward(self, state):
#         x = torch.relu(self.h_linear_1(state))
#         x = torch.relu(self.h_linear_2(x))
#         x = torch.relu(self.h_linear_3(x))
#         x = torch.tanh(self.h_linear_4(x))
#         return x
class Actor(nn.Module):

    def __init__(self, obs_dim, action_dim, learning_rate):
        super(Actor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 512)
        self.linear2 = nn.Linear(512, 128)
        self.linear3 = nn.Linear(128, self.action_dim)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, obs):
        x = F.relu(self.linear1(obs))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x
