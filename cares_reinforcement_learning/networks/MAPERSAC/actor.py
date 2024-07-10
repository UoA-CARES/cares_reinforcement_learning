"""
This is a stub file for the Actor class - reads directly off SAC's Actor class.
"""

# pylint: disable=unused-import

from cares_reinforcement_learning.networks.SAC import Actor as SACActor


class Actor(SACActor):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
        log_std_bounds: list[int] = None,
    ):
        if hidden_size is None:
            hidden_size = [400, 300]
        if log_std_bounds is None:
            log_std_bounds = [-20, 2]

        super().__init__(
            observation_size,
            num_actions,
            hidden_size=hidden_size,
            log_std_bounds=log_std_bounds,
        )
