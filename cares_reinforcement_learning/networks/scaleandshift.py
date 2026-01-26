import torch
from torch import nn
from cares_reinforcement_learning.util import helpers as hlp


class ScaleAndShift(nn.Module):
    def __init__(self):
        super().__init__()
        self.scales: torch.Tensor = torch.ones(1, device=hlp.get_device())
        self.shifts: torch.Tensor = torch.zeros(1, device=hlp.get_device())


    def set_film_parameters(self, scales: torch.Tensor, shifts: torch.Tensor):
        self.scales = scales  + torch.ones_like(scales) # center scaling around one
        self.shifts = shifts


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.scales.unsqueeze(1) + self.shifts.unsqueeze(1)