import torch
from torch import nn
from torch.nn import functional as F

from cares_reinforcement_learning.util.common import SquashedNormal


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int):
        super().__init__()
        self.hidden_size = [400, 300]
        self.log_std_bounds = [-20, 2]

        # Two hidden layers, 256 on each
        self.linear1 = nn.Linear(observation_size, self.hidden_size[0])
        self.linear2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.mean_linear = nn.Linear(self.hidden_size[1], num_actions)
        self.log_std_linear = nn.Linear(self.hidden_size[1], num_actions)
        # self.apply(weight_init)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
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
