"""
Original Paper: https://arxiv.org/abs/1511.05952
"""

import torch

from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.PERTD3 import Actor, Critic
from cares_reinforcement_learning.util.configurations import PERTD3Config


class PERTD3(TD3):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PERTD3Config,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )
