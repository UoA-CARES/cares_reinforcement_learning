from util import MemoryBuffer

import torch
import numpy as np

from gym import Space, Env
from agents.Agent import Agent


class SACAgent(Agent):

    def __init__(self,
                 memory: MemoryBuffer,
                 gamma: float,
                 tau: float,
                 alpha: float,
                 actor_net: torch.nn.Module,
                 actor_net_target: torch.nn.Module,
                 critic_one: torch.nn.Module,
                 critic_one_target: torch.nn.Module,
                 critic_two: torch.nn.Module,
                 critic_two_target: torch.nn.Module,
                 act_space: Space,
                 env: Env):
        super().__init__(env, memory)

        self.memory = memory

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = actor_net
        self.target_actor = actor_net_target

        self.critic_one = critic_one
        self.target_critic_one = critic_one_target

        self.critic_two = critic_two
        self.target_critic_two = critic_two_target

        self.act_space = act_space

        self.learn_counter = 0

    def choose_action(self, state):
        mean, std_dev = self.actor(state)

        return np.random.normal(mean, std_dev, 1) * self.act_space.high

    def learn(self, batch_size):

        # Only begin learning when there's enough experience in buffer to sample from
        if len(self.memory.buffer) < batch_size:
            return

        self.learn_counter += 1

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.FloatTensor(np.asarray(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.LongTensor(dones)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            mean, std_dev = self.actor(next_states)
            # TODO: consider num of actions: action vector
            # TODO: use squashed gaussian method to sample
            next_actions = torch.normal(mean, std_dev, batch_size)

            target_q_values_one = self.target_critic_one(next_states, next_actions)
            target_q_values_two = self.target_critic_two(next_states, next_actions)

            target_q_values = torch.min(target_q_values_one, target_q_values_two)

            # TODO: decode the entropy term
            q_target = rewards + self.gamma * (1 - dones) * (target_q_values - self.alpha * torch.log(next_actions))

        q_values_one = self.critic_one(states, actions)
        q_values_two = self.critic_two(states, actions)

        # Update the Critic One
        critic_one_loss = self.critic_one.loss(q_values_one, q_target)

        self.critic_one.optimizer.zero_grad()
        critic_one_loss.backward()
        self.critic_one.optimizer.step()

        # Update Critic Two
        critic_two_loss = self.critic_two.loss(q_values_two, q_target)

        self.critic_two.optimizer.zero_grad()
        critic_two_loss.backward()
        self.critic_two.optimizer.step()



