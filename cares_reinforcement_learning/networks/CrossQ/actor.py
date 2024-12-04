import torch
from torch import nn

from cares_reinforcement_learning.util.batchrenorm import BatchRenorm1d
from cares_reinforcement_learning.util.common import SquashedNormal
from cares_reinforcement_learning.util.configurations import CrossQConfig


class BaseActor(nn.Module):
    def __init__(
        self,
        act_net: nn.Module,
        mean_linear: nn.Linear,
        log_std_linear: nn.Linear,
        num_actions: int,
        log_std_bounds: list[float] | None = None,
    ):
        super().__init__()
        if log_std_bounds is None:
            log_std_bounds = [-20, 2]

        self.log_std_bounds = log_std_bounds

        self.num_actions = num_actions
        self.act_net = act_net

        self.mean_linear = mean_linear
        self.log_std_linear = log_std_linear

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


class DefaultActor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):

        hidden_sizes = [256, 256]
        log_std_bounds = [-20.0, 2.0]

        momentum = 0.01
        act_net = nn.Sequential(
            BatchRenorm1d(observation_size, momentum=momentum),
            nn.Linear(observation_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[1], momentum=momentum),
        )

        mean_linear = nn.Linear(hidden_sizes[1], num_actions)
        log_std_linear = nn.Linear(hidden_sizes[1], num_actions)

        super().__init__(
            act_net=act_net,
            mean_linear=mean_linear,
            log_std_linear=log_std_linear,
            num_actions=num_actions,
            log_std_bounds=log_std_bounds,
        )


class Actor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: CrossQConfig,
    ):
        hidden_sizes = config.hidden_size_actor
        log_std_bounds = config.log_std_bounds

        momentum = 0.01
        act_net = nn.Sequential(
            BatchRenorm1d(observation_size, momentum=momentum),
            nn.Linear(observation_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[1], momentum=momentum),
        )

        mean_linear = nn.Linear(hidden_sizes[1], num_actions)
        log_std_linear = nn.Linear(hidden_sizes[1], num_actions)

        super().__init__(
            act_net=act_net,
            mean_linear=mean_linear,
            log_std_linear=log_std_linear,
            num_actions=num_actions,
            log_std_bounds=log_std_bounds,
        )
