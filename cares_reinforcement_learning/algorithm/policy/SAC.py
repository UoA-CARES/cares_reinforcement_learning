"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py.

This code runs automatic entropy tuning
"""

import os
from collections import defaultdict
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F


class SAC:
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 device):

        self.type = "policy"
        self.actor_net = actor_network.to(device)  # this may be called policy_net in other implementations
        self.critic_net = critic_network.to(device)  # this may be called soft_q_net in other implementations

        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0
        self.policy_update_freq = 1

        self.device = device

        self.target_entropy = -action_num
        # self.target_entropy = -torch.prod(torch.Tensor([action_num]).to(self.device)).item()

        init_temperature = 0.01
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

    def select_action_from_policy(self, state, evaluation=False):
        # note that when evaluating this algorithm we need to select mu as action so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                action, _, _, = self.actor_net.sample(state_tensor)
            else:
                _, _, action, = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train_policy(self, experiences):
        self.learn_counter += 1
        info = {}

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net.sample(next_states)
            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two) - self.alpha * next_log_pi

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net.optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net.optimiser.step()

        pi, log_pi, _ = self.actor_net.sample(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net.optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net.optimiser.step()

        # update the temperature
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        info['q_target'] = q_target
        info['q_values_one'] = q_values_one
        info['q_values_two'] = q_values_two
        info['q_values_min'] = torch.minimum(q_values_one, q_values_two)
        info['critic_loss_total'] = critic_loss_total
        info['critic_loss_one'] = critic_loss_one
        info['critic_loss_two'] = critic_loss_two
        info['actor_loss'] = actor_loss
        
        return info

    def save_models(self, filename, filepath='models'):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f'{path}/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'{path}/{filename}_critic.pht')
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != 'models' else filepath

        self.actor_net.load_state_dict(torch.load(f'{path}/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'{path}/{filename}_critic.pht'))
        logging.info("models has been loaded...")
