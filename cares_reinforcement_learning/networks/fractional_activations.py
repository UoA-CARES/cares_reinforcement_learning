import torch
import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F

class FractionalReLUPositive(nn.Module):
    def __init__(self, a=0.1, epsilon=1e-6, clip_value=1.0):
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
class FPReLU2(nn.Module):
    def __init__(self, a=0.2, parameter_init=1.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)  # fix a
        self.p_raw = nn.Parameter(torch.tensor(parameter_init))  # learnable
        self.g = float(np.math.gamma(2 - float(a)))

    def p(self):
        # positive, smooth, and not huge; adjust shift if needed
        return F.softplus(self.p_raw)

    def forward(self, x):
        # Safe masks + exact f(0)=0; stable backprop near zero
        pos = (x > 0).float() * (x.clamp_min(1e-8) ** (1 - self.a)) / self.g
        neg = (x < 0).float() * (-(self.p()) / self.g) * ((-x).clamp_min(1e-8) ** (1 - self.a))
        return pos + neg

class FParReLU(nn.Module):
    def __init__(self, a=0.1, k_init=1.5):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)   # fixed exponent
        self.k = nn.Parameter(torch.tensor(k_init), requires_grad=True)  # learnable slope
        self.g = np.math.gamma(2 - float(a))  # normalization factor

    def forward(self, x):
        eps = 1e-8  # small value for numerical stability
        pos_mask = (x > 0).float()
        neg_mask = (x < 0).float()

        # Positive side (x > 0)
        positive_side = pos_mask * (1.0 / self.g) * (torch.clamp(x, min=eps) ** (1 - self.a))

        # Negative side (x < 0) -> k >= 1 makes it parametric
        negative_side = neg_mask * (-(torch.clamp(self.k, 0.5, 3.0) / self.g) *
                                    (torch.clamp(-x, min=eps) ** (1 - self.a)))

        # Zero stays zero
        return positive_side + negative_side

class FLReLU2(nn.Module):
    def __init__(self, a=0.1, k_init=0.1):
        super().__init__()
        self.a = nn.Parameter(torch.tensor(a), requires_grad=False)  # fixed exponent
        self.k = nn.Parameter(torch.tensor(k_init), requires_grad=True)  # learnable slope
        self.g = np.math.gamma(2 - float(a))

    def forward(self, x):
        pos_mask = (x > 0).float()
        neg_mask = (x < 0).float()

        # Positive side (x > 0)
        positive_side = pos_mask * (1.0 / self.g) * (torch.clamp(x, min=1e-8) ** (1 - self.a))

        # Negative side (x < 0)
        negative_side = neg_mask * (-(torch.clamp(self.k, 0.01, 0.3) / self.g) *
                                    (torch.clamp(-x, min=1e-8) ** (1 - self.a)))

        # For x == 0 → output = 0
        return positive_side + negative_side

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

class FLReLU(nn.Module):
    def __init__(self, a=0.1, epsilon=1e-6, clip_value=1.0):
        super().__init__()
        self.a = a
        self.epsilon = epsilon  # small value to replace zero input
        self.clip_value = clip_value
        self.k = 1.5
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
