import torch
from torch import nn
from torch.nn import functional as F

from cares_reinforcement_learning.util.common import SquashedNormal


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.hidden_size = [256, 256]
        self.log_std_bounds = [-20, 2]

        self.act_net = nn.Sequential(
            nn.Linear(state_dim, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(self.hidden_size[1], action_dim)
        self.log_std_linear = nn.Linear(self.hidden_size[1], action_dim)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act_net(state)
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Bound the action to finite interval.
        # Apply an invertible squashing function: tanh
        # employ the change of variables formula to compute the likelihoods of the bounded actions

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean
