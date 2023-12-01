import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch import distributions as pyd
import math


# class Actor(nn.Module):
#     def __init__(self, observation_size, num_actions):
#         super(Actor, self).__init__()
#
#         self.hidden_size = [1024, 1024]
#         self.log_sig_min = -20
#         self.log_sig_max = 2
#
#         self.h_linear_1 = nn.Linear(in_features=observation_size, out_features=self.hidden_size[0])
#         self.h_linear_2 = nn.Linear(in_features=self.hidden_size[0], out_features=self.hidden_size[1])
#
#         self.mean_linear    = nn.Linear(in_features=self.hidden_size[1], out_features=num_actions)
#         self.log_std_linear = nn.Linear(in_features=self.hidden_size[1], out_features=num_actions)
#
#     def forward(self, state):
#         x = F.relu(self.h_linear_1(state))
#         x = F.relu(self.h_linear_2(x))
#
#         mean    = self.mean_linear(x)
#         log_std = self.log_std_linear(x)
#         log_std = torch.clamp(log_std, min=self.log_sig_min, max=self.log_sig_max)
#
#         return mean, log_std
#
#     def sample(self, state):
#         mean, log_std = self.forward(state)
#         std    = log_std.exp()
#         normal = Normal(mean, std)
#
#         x_t    = normal.rsample()  # for re-parameterization trick (mean + std * N(0,1))
#         y_t    = torch.tanh(x_t)
#         action = y_t
#
#         epsilon  = 1e-6
#         log_prob = normal.log_prob(x_t)
#         log_prob -= torch.log((1 - y_t.pow(2)) + epsilon)
#         log_prob = log_prob.sum(1, keepdim=True)
#         mean     = torch.tanh(mean)
#
#         return action, log_prob, mean


class TanhTransform(pyd.transforms.Transform):
    r"""
    Transform via the mapping :math:`y = \tanh(x)`.
    It is equivalent to
    ```
    ComposeTransform([AffineTransform(0., 2.), SigmoidTransform(), AffineTransform(-1., 2.)])
    ```
    However this might not be numerically stable, thus it is recommended to use `TanhTransform`
    instead.
    Note that one should use `cache_size=1` when it comes to `NaN/Inf` values.
    """
    domain = pyd.constraints.real
    codomain = pyd.constraints.interval(-1.0, 1.0)
    bijective = True
    sign = +1

    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, TanhTransform)

    def _call(self, x):
        return x.tanh()

    def _inverse(self, y):
        print("--------------------------------------")
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)

    def log_abs_det_jacobian(self, x, y):
        # This function is often used to compute the log
        # We use a formula that is more numerically stable, see details in the following link
        # https://github.com/tensorflow/probability/commit/ef6bb176e0ebd1cf6e25c6b5cecdd2428c22963f#diff-e120f70e92e6741bca649f04fcd907b7
        return 2. * (math.log(2.) - x - F.softplus(-2. * x))


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)
        # a = tanh(u)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.hidden_size = [256, 256]
        self.log_std_bounds = [-20, 2]
        # Two hidden layers, 256 on each
        self.linear1 = nn.Linear(state_dim, self.hidden_size[0])
        self.linear2 = nn.Linear(self.hidden_size[0], self.hidden_size[1])
        self.mean_linear = nn.Linear(self.hidden_size[1], action_dim)
        self.log_std_linear = nn.Linear(self.hidden_size[1], action_dim)
        # self.apply(weight_init)

    def sample(self, obs):
        x = F.relu(self.linear1(obs))
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
