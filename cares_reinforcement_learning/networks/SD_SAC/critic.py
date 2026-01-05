from cares_reinforcement_learning.networks.SACD import Critic as SACDCritic
from cares_reinforcement_learning.util.configurations import SD_SACConfig
from cares_reinforcement_learning.networks.common import MLP

class Critic(SACDCritic):
    def __init__(self, observation_size: int, action_num: int, config: SD_SACConfig, encoder_net: MLP):
        super().__init__(observation_size, action_num, config, encoder_net)