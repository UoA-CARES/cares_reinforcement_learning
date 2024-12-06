"""
Original Paper: https://openreview.net/pdf?id=xCVJMsPv3RT
Code based on: https://github.com/TakuyaHiraoka/Dropout-Q-Functions-for-Doubly-Efficient-Reinforcement-Learning/blob/main/KUCodebase/code/agent.py

This code runs automatic entropy tuning
"""

import torch

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.DroQ import Actor, Critic
from cares_reinforcement_learning.util.configurations import DroQConfig


class DroQ(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: DroQConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)
