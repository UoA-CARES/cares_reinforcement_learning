"""
Original Paper: https://arxiv.org/abs/1511.05952
"""

import torch

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.PERSAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import PERSACConfig


class PERSAC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PERSACConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)
