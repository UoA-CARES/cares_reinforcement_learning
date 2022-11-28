from ..util import MemoryBuffer, Plotter, train
from ..networks import Actor, Critic
from ..agents import DDPGAgent

import torch
from gym import Env
import numpy as np


class DDPG:

    def __init__(self,
                 env: Env,
                 memory_capacity: int,
                 batch_size: int,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 gamma: float,
                 tau: float,
                 episode_num: int):

        self.env = env

        actor_net = Actor(
            obs_space=env.observation_space,
            act_space=env.action_space,
            learning_rate=learning_rate_actor
        )

        actor_net_target = Actor(
            obs_space=env.observation_space,
            act_space=env.action_space,
            learning_rate=learning_rate_actor
        )

        critic_net = Critic(
            observation_size=env.observation_space.shape[0],
            num_actions=env.action_space.shape[0],
            learning_rate=learning_rate_critic
        )

        critic_net_target = Critic(
            observation_size=env.observation_space.shape[0],
            num_actions=env.action_space.shape[0],
            learning_rate=learning_rate_critic
        )

        memory = MemoryBuffer(memory_capacity)

        self.agent = DDPGAgent(
            env=env,
            memory=memory,
            gamma=gamma,
            tau=tau,
            actor_net=actor_net,
            target_actor_net=actor_net_target,
            critic_net=critic_net,
            target_critic_net=critic_net_target,
        )

        self.ep_num = episode_num
        self.batch_size = batch_size

    def train(self):
        self.agent.fill_buffer()
        reward_data = train(self.agent, self.ep_num, self.batch_size, self.env)

        return reward_data
