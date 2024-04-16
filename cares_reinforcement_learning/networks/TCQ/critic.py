import torch
from torch import nn

from cares_reinforcement_learning.networks.TCQ import Mlp


class Critic(nn.Module):
    def __init__(self, observation_size, num_actions, num_quantiles, num_nets):
        super().__init__()

        self.nets = []
        self.n_quantiles = num_quantiles
        self.n_nets = num_nets

        for i in range(num_nets):
            net = Mlp(observation_size + num_actions, [512, 512, 512], num_quantiles)
            self.add_module(f"qf{i}", net)
            self.nets.append(net)

    def forward(self, state, action):
        sa = torch.cat((state, action), dim=1)
        quantiles = torch.stack(tuple(net(sa) for net in self.nets), dim=1)
        return quantiles
