import torch
import torch.nn as nn
import torch.nn.functional as F

from cares_reinforcement_learning.util.configurations import TD7Config


def AvgL1Norm(x, eps=1e-8):
    return x / x.abs().mean(-1, keepdim=True).clamp(min=eps)


class Encoder(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):
        super(Encoder, self).__init__()

        self.activ = F.elu
        hdim = 256
        zs_dim = config.zs_dim
        state_dim = observation_size
        action_dim = num_actions

        # state encoder
        self.zs1 = nn.Linear(state_dim, hdim)
        self.zs2 = nn.Linear(hdim, hdim)
        self.zs3 = nn.Linear(hdim, zs_dim)

        # state-action encoder
        self.zsa1 = nn.Linear(zs_dim + action_dim, hdim)
        self.zsa2 = nn.Linear(hdim, hdim)
        self.zsa3 = nn.Linear(hdim, zs_dim)

    def zs(self, state):
        zs = self.activ(self.zs1(state))
        zs = self.activ(self.zs2(zs))
        zs = AvgL1Norm(self.zs3(zs))
        return zs

    def zsa(self, zs, action):
        zsa = self.activ(self.zsa1(torch.cat([zs, action], 1)))
        zsa = self.activ(self.zsa2(zsa))
        zsa = self.zsa3(zsa)
        return zsa
