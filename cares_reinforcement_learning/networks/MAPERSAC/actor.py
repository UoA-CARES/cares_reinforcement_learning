"""
This is a stub file for the Actor class - reads directly off SAC's Actor class.
"""

# pylint: disable-next=unused-import
from cares_reinforcement_learning.networks.SAC import Actor
from cares_reinforcement_learning.networks.SAC import DefaultActor as SACDefaultActor


class DefaultActor(SACDefaultActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        log_std_bounds = [-20.0, 2.0]

        hidden_sizes = [400, 300]

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            hidden_sizes=hidden_sizes,
            log_std_bounds=log_std_bounds,
        )
