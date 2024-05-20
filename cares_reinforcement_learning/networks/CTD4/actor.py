from cares_reinforcement_learning.networks.TD3 import Actor as TD3Actor


class Actor(TD3Actor):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ):
        super().__init__(observation_size, num_actions, hidden_size)
