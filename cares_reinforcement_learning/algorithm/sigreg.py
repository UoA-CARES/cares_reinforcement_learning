"""
SIGReg (Sketched Isotropic Gaussian Regularisation)
---------------------------------------------------

Reference: LeJEPA - https://arxiv.org/abs/2511.08544

Pushes a batch of embeddings towards an isotropic Gaussian distribution by
comparing the empirical characteristic function of random 1D projections
against that of a standard normal (Epps-Pulley statistic), integrated with
the trapezoidal rule over t in [0, t_max].
"""

import torch
from torch import nn


class SIGReg(nn.Module):
    def __init__(self, knots: int = 17, t_max: float = 3.0, num_slices: int = 256):
        super().__init__()

        self.num_slices = num_slices

        t = torch.linspace(0, t_max, knots, dtype=torch.float32)
        dt = t_max / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            proj: embeddings of shape (..., N, D) where N is the batch dimension.

        Returns:
            Scalar SIGReg statistic averaged over random projection slices.
        """
        # Random unit-norm projection directions (D, num_slices)
        A = torch.randn(  # pylint: disable=invalid-name
            proj.size(-1), self.num_slices, device=proj.device, dtype=proj.dtype
        )
        A = A.div_(A.norm(p=2, dim=0))  # pylint: disable=invalid-name

        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()
