from cares_reinforcement_learning.networks.SAC import Actor as SACActor


class Actor(SACActor):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
        log_std_bounds: list[int] = None,
    ):
        super().__init__(observation_size, num_actions, hidden_size, log_std_bounds)
