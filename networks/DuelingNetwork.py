import torch
import torch.nn as nn
import torch.optim as optim

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")


class DuelingNetwork(nn.Module):

    def __init__(self, observation_space_size, action_num, learning_rate):
        super(DuelingNetwork, self).__init__()
        self.input_dim = observation_space_size
        self.output_dim = action_num

        self.feature_layer = nn.Sequential(
            nn.Linear(self.input_dim, 128),
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

    def forward(self, state):
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals