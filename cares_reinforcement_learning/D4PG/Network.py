#Both actor and critic Network_d4pg
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as nn

class Network(nn.Module):
    def __init__(self, observation_size, action_size, seed, h_linear_1, h_linear_2):

        super(Network, self).__init__()

        self.hidden_size = [1024, 1024]

        self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
        self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
        self.h_linear_3 = nn.Linear(in_features=self.hidden_size[1], out_features=action_size)

        self.optimiser = optim.Adam(self.parameters(), lr=learning_rate)
    
        self.action_size = action_size
        self.state_size = observation_size 
        #self.seed = random.seed(random_seed)

    def forward(self, state):
        x = nn.relu(self.h_linear_1(state))
        x = nn.relu(self.h_linear_2(x))
        x = self.h_linear_3(x)
        return x
