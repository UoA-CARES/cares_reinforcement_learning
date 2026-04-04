import torch
import torch.nn as nn
'''
This is a rough draft of what I think the observation normalization layer should do.

It sets the mean at 0, and the variance at one using the torch library.
When self training, it includes a slight historical bias (not sure if thats the right term).
'''

class RunningNormalization(nn.Module):
    def __init__(self, obs_size):
        self.register_buffer("mean", torch.zeros(obs_size))
        self.register_buffer("var",  torch.ones(obs_size))

    def forward(self, x):
        if self.training:
            self.mean = 0.9 * self.mean + 0.1 * x.mean(0)
            self.variance  = 0.9 * self.variance  + 0.1 * x.var(1)
        return ((x - self.mean) / (self.variance.sqrt() + 1e-8))

