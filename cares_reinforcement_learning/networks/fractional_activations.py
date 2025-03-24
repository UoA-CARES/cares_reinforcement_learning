import torch
import torch.nn as nn


class FractionalReLU(nn.Module):

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        output = torch.where(x > 0, torch.pow(x, 1 - self.a), torch.zeros_like(x))
        print(output)
        return output


# Fractional ReLU Gamma version activation function
class FractionalReLUGamma(nn.Module):

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        gamma_factor = 1 / torch.special.gamma(torch.tensor(2 - self.a, dtype=x.dtype))
        return torch.where(
            x > 0,
            gamma_factor * torch.pow(torch.abs(x), 1 - self.a),
            torch.zeros_like(x),
        )


# Fractional ReLU Custom version2 activation function
class FractionalReLUCustom(nn.Module):

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        return torch.where(
            x > 0,
            (2 - self.a) + torch.pow(x, 1 - self.a) * (self.a - 2) * (1 - self.a),
            torch.zeros_like(x),
        )


# Fractional Tanh activation function
class FractionalTanh(nn.Module):
    """Fractional-order Tanh activation."""

    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a

    def forward(self, x):
        # Adding small epsilon to avoid zero power issues
        return torch.tanh(x) * torch.pow(torch.abs(x) + 1e-6, 1 - self.a)
