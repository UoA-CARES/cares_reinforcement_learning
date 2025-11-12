from cares_reinforcement_learning.networks.SACD import Actor as SACDActor
from cares_reinforcement_learning.util.configurations import SD_SACConfig

class Actor(SACDActor):
    def __init__(self, config: SD_SACConfig, state_dim: int, action_dim: int):
        super().__init__(config, state_dim, action_dim)