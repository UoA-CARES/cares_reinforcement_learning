"""
Original Paper: https://doi.org/10.1609/aaai.v33i01.33014213
"""

import torch

from cares_reinforcement_learning.algorithm.policy import MADDPG
from cares_reinforcement_learning.algorithm.policy.DDPG import DDPG
from cares_reinforcement_learning.util.configurations import M3DDPGConfig


class M3DDPG(MADDPG):
    def __init__(
        self,
        agents: list[DDPG],
        config: M3DDPGConfig,
        device: torch.device,
    ):
        super().__init__(agents, config, device)
