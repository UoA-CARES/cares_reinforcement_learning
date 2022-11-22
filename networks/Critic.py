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


# class Critic(nn.Module):
#     def __init__(self, observation_size, num_actions, learning_rate):
#         super(Critic, self).__init__()
#
#         self.vector_size = observation_size
#         self.num_actions = num_actions
#         self.input_size = self.vector_size + self.num_actions
#         self.hidden_size = [128, 64, 32]
#
#         self.Q1 = nn.Sequential(
#             nn.Linear(self.input_size, self.hidden_size[0]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size[0], self.hidden_size[1]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size[1], self.hidden_size[2]),
#             nn.ReLU(),
#             nn.Linear(self.hidden_size[2], 1)
#         )
#
#         self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#         self.loss = nn.MSELoss()
#
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         q1 = self.Q1(x)
#         return q1

class Critic(nn.Module):

    def __init__(self, obs_dim, action_dim, learning_rate):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim

        self.linear1 = nn.Linear(self.obs_dim, 1024)
        self.linear2 = nn.Linear(1024 + self.action_dim, 512)
        self.linear3 = nn.Linear(512, 300)
        self.linear4 = nn.Linear(300, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x, a):
        x = F.relu(self.linear1(x))
        xa_cat = torch.cat([x,a], 1)
        xa = F.relu(self.linear2(xa_cat))
        xa = F.relu(self.linear3(xa))
        qval = self.linear4(xa)

        return qval