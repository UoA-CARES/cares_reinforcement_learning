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


class Network(nn.Module):

    def __init__(self, observation_space_size, action_num, learning_rate):
        super().__init__()

        self.fc1 = nn.Linear(observation_space_size, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, action_num)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x