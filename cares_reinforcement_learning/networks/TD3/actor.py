from cares_reinforcement_learning.networks.DDPG.actor import Actor as DDPGActor


class Actor(DDPGActor):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
    ):
        if hidden_size is None:
            hidden_size = [256, 256]
        super().__init__(observation_size, num_actions, hidden_size)
