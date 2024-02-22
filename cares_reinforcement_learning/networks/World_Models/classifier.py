import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, observation_size):
        super().__init__()
        self.linear1 = nn.Linear(observation_size, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 1)

    def forward(self, state):
        """
        Forward
        :param state:
        :return:
        """
        x_1 = self.linear1(state)
        x_1 = F.leaky_relu(x_1, negative_slope=0.2, inplace=True)
        x_1 = self.linear2(x_1)
        x_1 = F.leaky_relu(x_1, negative_slope=0.2, inplace=True)
        x_1 = self.linear3(x_1)
        x_1 = F.sigmoid(x_1)
        return x_1