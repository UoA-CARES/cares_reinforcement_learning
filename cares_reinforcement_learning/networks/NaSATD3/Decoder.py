import torch
import torch.nn as nn
from cares_reinforcement_learning.networks.NaSATD3.weight_initialization import (
    weight_init,
)


class Decoder(nn.Module):
    def __init__(self, latent_dim, k=9):
        super(Decoder, self).__init__()
        self.num_filters = 32
        self.latent_dim = latent_dim

        self.fc_1 = nn.Linear(self.latent_dim, 39200)

        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=self.num_filters,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=self.num_filters,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=self.num_filters,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=k,
                kernel_size=3,
                stride=2,
                output_padding=1,
            ),
            nn.Sigmoid(),
        )

        self.apply(weight_init)

    def forward(self, x):
        x = torch.relu(self.fc_1(x))
        x = x.view(-1, 32, 35, 35)
        x = self.deconvs(x)
        return x
