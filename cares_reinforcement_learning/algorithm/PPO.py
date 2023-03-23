"""
Original Paper: https://arxiv.org/abs/1707.06347
Good Explanation: https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on: https://github.com/ericyangyu/PPO-for-Beginners


"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F


class PPO:
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 action_num,
                 device):

        self.actor_net  = actor_network.to(device)
        self.critic_net = critic_network.to(device)


        self.gamma = gamma

        self.action_num = action_num
        self.device = device

    def train_policy(self):
        pass
