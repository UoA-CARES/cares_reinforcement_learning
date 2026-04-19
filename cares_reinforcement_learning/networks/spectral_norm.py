import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm

class SpectralNormLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True, 
                 n_power_iterations=1, eps=1e-12, dim=0):
        super().__init__()
        
        # Create a standard Linear layer
        linear = nn.Linear(in_features, out_features, bias=bias)
        
        # Apply spectral normalization parametrization
        self.linear = spectral_norm(
            linear,
            name="weight",
            n_power_iterations=n_power_iterations,
            eps=eps,
            dim=dim
        )

    def forward(self, x):
        return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias