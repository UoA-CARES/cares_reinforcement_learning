import torch
import torch.nn as nn
import torch
import numpy as np


class FractionalReLUPositive(nn.Module):
    def __init__(self, a=0.5, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.clip_value = clip_value

    def forward(self, x):
        # print(f"input: {x}")
        epsilon = 1e-6
        x_clamped = torch.clamp(x, min=epsilon)  # Ensure no zero values in the negative branch
        
        # Apply fractional power to positive and negative inputs
        positive_part = torch.pow(x_clamped, 1 - self.a)
        negative_part =  0 #torch.ones_like(x) * self.epsilon
        
        # Apply scaled positive and negative parts
        output = torch.where(x > 0, positive_part,  negative_part)
        return output

class FractionaLLeakyReLU(nn.Module):
    def __init__(self, a=0.5, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon
        self.clip_value = clip_value

    def forward(self, x):
        # print(f"input: {x}")
        epsilon = 1e-6
        x_clamped = torch.clamp(x, min=epsilon)  # Ensure no zero values in the negative branch
        
        # Compute positive and negative sides element-wise
        positive_side = (1 / np.math.gamma(2 - self.a)) * torch.where(x_clamped > 0, x_clamped ** (1 - self.a), torch.tensor(self.epsilon, device=x.device))
        negative_side = (0.1 / np.math.gamma(2 - self.a)) * torch.where(x_clamped <= 0, torch.abs(x_clamped) ** (1 - self.a), torch.tensor(self.epsilon, device=x.device))
    
        return positive_side + negative_side
class FractionaLLeakyReLU2(nn.Module):
    def __init__(self, a=0.5, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon  # small value to replace zero input
        self.clip_value = clip_value
        self.k = 0.1
        self.g = np.math.gamma(2 - a)

    def forward(self, x):
        # Positive side: x > 0
        positive_side = torch.where(x > 0, (1.0 / self.g) * x ** (1 - self.a),
                                    (1.0 / self.g) * torch.full_like(x, self.epsilon) * (x == 0))

        # Negative side: x < 0
        negative_side = torch.where(x < 0, - (self.k / self.g) * (-x) ** (1 - self.a),
                                    - (self.k / self.g) * torch.full_like(x, self.epsilon) * (x == 0))

        return positive_side + negative_side

class FractionalPReLU(nn.Module):
    def __init__(self, alpha=0.5, epsilon=1e-6):
        super().__init__()
        self.alpha = alpha
        self.epsilon = epsilon  # small value for x=0
        self.a = 1.5  # negative slope
        self.g = np.math.gamma(2 - alpha)

    def forward(self, x):
        # Positive side: x > 0
        positive_side = torch.where(x > 0,
                                    (1.0 / self.g) * x ** (1 - self.alpha),
                                    (1.0 / self.g) * torch.full_like(x, self.epsilon) * (x == 0))
        # Negative side: x < 0
        negative_side = torch.where(x < 0,
                                    - (self.a / self.g) * (-x) ** (1 - self.alpha),
                                    - (self.a / self.g) * torch.full_like(x, self.epsilon) * (x == 0))

        return positive_side + negative_side

        
        
class ReLU(nn.Module):
    def __init__(self, a: float = 0.5):
        super().__init__()
        self.a = a
        self.counter = 0
    def forward(self, x):
        output = nn.functional.relu(x).pow(self.a)
        print(f"input: {x}")
        return output


# class FractionalLeakyReLU(nn.Module):
#     def __init__(self, a: float = 0.98):
#         super().__init__()
#         self.a = a
        
#         # Pre-compute the gamma value since it's a constant for each forward pass
#         self.gamma_value = torch.exp(torch.special.gammaln(torch.tensor(2.0 - self.a)))
        
#     def forward(self, x):
#         # Clamp input to avoid issues with extremely small values
#         epsilon = 1e-6
#         x_clamped = torch.clamp(x, min=epsilon)  # Ensure no zero values in the negative branch
        
#         # Apply fractional power to positive and negative inputs
#         positive_part = torch.pow(x_clamped, 1 - self.a)
#         negative_part = torch.pow(torch.abs(x), 1 - self.a)
        
#         # Apply scaled positive and negative parts
#         output = torch.where(x > 0, 
#                              (1 / self.gamma_value) * positive_part,  
#                              (0.1 / self.gamma_value) * negative_part)
        
#         # Handle NaNs in the output
#         output = torch.nan_to_num(output, nan=0.0)  # Replace NaN with 0, can replace with a different value if needed
#         self.counter += 1
#         #print(f"counter: {self.counter}")
#         # print(f"input: {x}")
#         # print(f"output: {output}")
        
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