
import torch
import torch.nn as nn
from cares_reinforcement_learning.networks.NaSATD3.weight_initialization import weight_init

class Encoder(nn.Module):
    def __init__(self, latent_dim, k=9):
        super(Encoder, self).__init__()
        self.num_layers  = 4
        self.num_filters = 32
        self.latent_dim  = latent_dim

        self.cov_net = nn.ModuleList([nn.Conv2d(k, self.num_filters, 3, stride=2)])
        for i in range(self.num_layers - 1):
            self.cov_net.append(nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1))

        self.fc = nn.Linear(39200, self.latent_dim)
        self.ln = nn.LayerNorm(self.latent_dim)

        self.apply(weight_init)

    def forward_conv(self, x):
        conv = torch.relu(self.cov_net[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.cov_net[i](conv))
        h = torch.flatten(conv, start_dim=1)
        return h

    def forward(self, obs, detach=False):
        h      = self.forward_conv(obs)
        h_fc   = self.fc(h)
        h_norm = self.ln(h_fc)
        out    = torch.tanh(h_norm)
        if detach:
            out = out.detach()
        return out
