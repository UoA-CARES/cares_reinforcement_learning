from ..util import MemoryBuffer

import torch
import numpy as np

from gym import Space


class TD3Agent:

    def __init__(self,
                 memory: MemoryBuffer,
                 gamma: float,
                 tau: float,
                 actor_net: torch.nn.Module,
                 actor_net_target: torch.nn.Module,
                 critic_one: torch.nn.Module,
                 critic_one_target: torch.nn.Module,
                 critic_two: torch.nn.Module,
                 critic_two_target: torch.nn.Module,
                 act_space: Space
                 ):
        """
                Constructor used to create DDPGAgent

                Input:
                    `memory`: buffer that stores experience \n
                    `gamma`: discount rate \n
                    `tau`: pol yak averaging constant, lagging constant \n
                    `actor_net`: Neural Network used to approximate the policy \n
                    `actor_net_target`: Lagging Neural Network used to control over estimation \n
                    `critic_one`: Neural Network approximating the Q function, used to critique the policy \n
                    `critic_one_target`: Lagging Neural Network used to control over estimation \n
                    `critic_two`: Neural Network approximating the Q function, used to critique the policy \n
                    `critic_two_target`: Lagging Neural Network used to control over estimation \n
        """

        self.actor = actor_net
        self.actor_target = actor_net_target

        self.critic_one = critic_one
        self.critic_one_target = critic_one_target

        self.critic_two = critic_two
        self.critic_two_target = critic_two_target

        self.memory = memory

        self.gamma = gamma
        self.tau = tau

        self.act_space = act_space

        self.policy_update_freq = 3
        self.learn_counter = 0

    def choose_action(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy()

        noise = np.random.normal(0, scale=0.1 * self.act_space.high.max(), size=self.act_space.shape[0])
        clipped_action = np.clip(action[0] + noise, self.act_space.low, self.act_space.high)

        return clipped_action

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

            next_actions = self.actor_target(next_states)

            noise = 0.2 * torch.rand_like(next_actions)
            clipped_noise = noise.clamp(-0.5, 0.5)

            next_actions = torch.clip(next_actions + clipped_noise, torch.FloatTensor(self.act_space.low), torch.FloatTensor(self.act_space.high))

            target_q_values_one = self.critic_one_target(next_states, next_actions)
            target_q_values_two = self.critic_two_target(next_states, next_actions)

            target_q_values = torch.min(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

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

        if self.learn_counter % self.policy_update_freq == 0:

            # Update Actor
            actor_q_one = self.critic_one(states, self.actor(states))
            actor_q_two = self.critic_two(states, self.actor(states))

            actor_q_values = torch.min(actor_q_one, actor_q_two)

            actor_loss = -actor_q_values.mean()

            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            # Update target network params
            for target_param, param in zip(self.critic_one_target.parameters(), self.critic_one.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_two_target.parameters(), self.critic_two.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
