"""
Custom weight init for Conv2D and Linear layers
need to move this function to the cares LR or cares lib folder

delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
"""

from torch import nn


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)

    elif isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain("relu")
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)
