import torch
from torch import Tensor, nn

class GoLU(nn.Module):
    """
    GoLU activation function. From https://arxiv.org/pdf/2502.03654
    """

    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        # GoLU(x) = x*Gompertz(x), where Gompertz(x) = e^(−e^−x)
        return input * torch.exp(-torch.exp(-input))
