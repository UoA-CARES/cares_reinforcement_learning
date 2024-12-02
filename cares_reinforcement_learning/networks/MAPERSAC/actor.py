"""
This is a stub file for the Actor class - reads directly off SAC's Actor class.
"""

from torch import nn

# pylint: disable-next=unused-import
from cares_reinforcement_learning.networks.SAC import Actor, BaseActor


class DefaultActor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        log_std_bounds = [-20.0, 2.0]

        hidden_sizes = [400, 300]

        act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
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
