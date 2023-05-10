"""
Original Paper: https://arxiv.org/abs/2007.06049
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F


class TD3_PER(object):
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 device):
        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num
        self.device = device

        # new ones
        self.alpha = 0.4
        self.min_priority = 1
        self.noise_clip = 0.5  # self.noise_clip = noise_clip
        self.policy_noise = 0.2  # self.policy_noise = policy_noise

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # todo check if really need this line
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()

            # this is part the TD3 too, add noise to the action
            noise = np.random.normal(0, scale=0.10, size=self.action_num)
            action = action + noise
            action = np.clip(action, -1, 1)
        return action

    # def train_policy(self, experiences):
    def train_policy(self, replay_buffer, batch_size):
        self.learn_counter += 1

        # Sample replay buffer
        states, actions, next_states, rewards, dones, ind, weights = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)  # torch.min

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        # Get current Q estimates way1
        q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_1 = (q_values_one - q_target).abs()
        critic_loss_2 = (q_values_two - q_target).abs()
        critic_loss_total = self.huber(critic_loss_1) + self.huber(critic_loss_2)

        # Optimize the critic
        self.critic_net.optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net.optimiser.step()

        priority = torch.max(critic_loss_1, critic_loss_2).clamp(min=self.min_priority).pow(
            self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority(ind, priority)

        # Get current Q estimates way2 (original PER)
        """q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_1 = (q_values_one - q_target).abs()
        critic_loss_2 = (q_values_two - q_target).abs()
        # Compute critic loss
        critic_loss_total = ((weights * F.mse_loss(q_values_one, q_target, reduction='none')).mean()
                           + (weights * F.mse_loss(q_values_two, q_target, reduction='none')).mean())

        # Optimize the critic
        self.critic_net.optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net.optimiser.step()

        priority = torch.max(critic_loss_1 , critic_loss_2 ).pow(self.alpha).cpu().data.numpy().flatten()
        replay_buffer.update_priority(ind, priority)"""

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_q_one, actor_q_two = self.critic_net(states, self.actor_net(states))
            actor_q_values = torch.minimum(actor_q_one, actor_q_two)
            actor_loss = -actor_q_values.mean()

            # Optimize the actor
            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net.optimiser.step()

            # Update target network params
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def huber(self, x):
        return torch.where(x < self.min_priority, 0.5 * x.pow(2), self.min_priority * x).mean()

    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor_net.state_dict(), f'models/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'models/{filename}_critic.pht')
        logging.info("models has been loaded...")

    def load_models(self, filename):
        self.actor_net.load_state_dict(torch.load(f'models/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'models/{filename}_critic.pht'))
        logging.info("models has been loaded...")
