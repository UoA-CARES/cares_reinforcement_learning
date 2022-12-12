from ..util import MemoryBuffer, Plotter, train
from ..networks import Network
from ..agents import DoubleDQNAgent

import torch
from gym import Env
import numpy as np


class DoubleDQN:

    def __init__(self,
                 env: Env,
                 memory_capacity: int,
                 batch_size: int,
                 learning_rate_main: float,
                 learning_rate_target: float,
                 gamma: float,
                 tau: float,
                 epsilon_max: float,
                 epsilon_min: float,
                 epsilon_decay: float,
                 episode_num: int):

        self.env = env

        obs_size = env.observation_space.shape[0]
        act_size = env.action_space.n

        main_net = Network(
            observation_space_size=obs_size,
            action_num=act_size,
            learning_rate=learning_rate_main
        )

        target_net = Network(
            observation_space_size=obs_size,
            action_num=act_size,
            learning_rate=learning_rate_target
        )

        memory = MemoryBuffer(memory_capacity)

        self.agent = DoubleDQNAgent(
            main_network=main_net,
            target_network=target_net,
            memory=memory,
            epsilon_info=(epsilon_max, epsilon_min, epsilon_decay),
            gamma=gamma,
            tau=tau,
            action_space=env.action_space
        )

        self.ep_num = episode_num
        self.batch_size = batch_size

    def train(self):

        reward_data = train(self.agent, self.ep_num, self.batch_size, self.env)

        return reward_data