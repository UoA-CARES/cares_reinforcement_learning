import torch
import torch.nn as nn


class GoLU(nn.Module):
    """
    GoLU activation function. From https://arxiv.org/pdf/2502.03654

    GoLU(x) = x * Gompertz(x)
    Gompertz(x) = a * exp(-b * exp(-c * x))

    Args:
        a (float): Controls the y-scale of the function. Default is 1.0.
        b (float): Controls the x-displacement of the gate close to the origin. Default is 1.0.
        c (float): Controls the growth rate of the gate. Default is 1.0.

    Note - Don't set alpha, beta and gamma to negative values, else the Gompertz gate looses its classical S-shape.
    """

    def __init__(self, a: float = 1.0, b: float = 1.0, c: float = 1.0):
        super().__init__()
        self.a = a
        self.b = b
        self.c = c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Prevent intermediate overflow for large negative inputs
        # In the original paper, this function is implemented in cuda and is allowed to overflow since it doesn't crash the program.
        x_safe = torch.clamp(x, min=-60.0)
        y = x * self.a * torch.exp(-self.b * torch.exp(-self.c * x_safe))

        # if torch.any(x < -60.0):
        #     print("Warning: GoLU input clamped to prevent overflow.")
        return y
