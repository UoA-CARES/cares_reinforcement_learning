import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

DEVICE = torch.device('cuda')

class DuelingNetwork(nn.Module):

    def __init__(self, input_dim, output_dim, learning_rate):
        super(DuelingNetwork, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.feauture_layer = nn.Sequential(
            nn.Linear(self.input_dim[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.output_dim)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, state):
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        
        return qvals

FC1_DIMS = 1024
FC2_DIMS = 512

class Network(nn.Module):
    
    def __init__(self, obs_space_shape, act_space_shape, learning_rate):
        super().__init__()
        
        self.fc1 = nn.Linear(*obs_space_shape, FC1_DIMS)
        self.fc2 = nn.Linear(FC1_DIMS, FC2_DIMS)
        self.fc3 = nn.Linear(FC2_DIMS, act_space_shape)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()
        self.to(DEVICE)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

class Actor(nn.Module):
    def __init__(self, vector_size, num_actions):
        super(Actor, self).__init__()

        self.input_size  = vector_size
        self.num_actions = num_actions
        self.hidden_size = [128, 64, 32]

        self.h_linear_1 = nn.Linear(in_features=self.input_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=self.hidden_size[2])
        self.h_linear_4 = nn.Linear(in_features=self.hidden_size[2], out_features=self.num_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self, state):
        x = torch.relu(self.h_linear_1(state))
        x = torch.relu(self.h_linear_2(x))
        x = torch.relu(self.h_linear_3(x))
        x = torch.tanh(self.h_linear_4(x))
        return x  

class Critic(nn.Module):
    def __init__(self, vector_size, num_actions):
        super(Critic, self).__init__()

        self.vector_size = vector_size
        self.num_actions = num_actions
        self.input_size  = self.vector_size + self.num_actions
        self.hidden_size = [128, 64, 32]

        self.Q1 = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], self.hidden_size[2]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[2], 1)  # no sure why is always one here
        )

        self.optimizer = optim.Adam(self.parameters(), lr=0.0001)
        self.loss = nn.MSELoss()
        self.to(DEVICE)

    def forward(self,  state, action):
        x = torch.cat([state, action], dim=1)
        q1 = self.Q1(x)
        return q1