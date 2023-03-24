
"""
Original Paper:
                https://arxiv.org/abs/1707.06347
Good Explanation:
                https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on:
                https://github.com/ericyangyu/PPO-for-Beginners
                https://github.com/nikhilbarhate99/PPO-PyTorch
"""

import os
import copy
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 action_num,
                 device):

        self.actor_net  = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma  = gamma
        self.device = device
        self.action_num = action_num

        self.k = 10
        self.eps_clip = 0.2
        self.cov_var  = torch.full(size=(action_num,), fill_value=0.5).to(self.device)
        self.cov_mat  = torch.diag(self.cov_var)

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)  # todo no sure about this check also fo the other algorithms, me quede aqui

            mean = self.actor_net(state_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)

            # Sample an action from the distribution and get its log prob
            action   = dist.sample()

            log_prob = dist.log_prob(action)

            action   = action.cpu().data.numpy().flatten()
            log_prob = log_prob.cpu().data.numpy().flatten()  # just to have this as numpy array

        return action, log_prob

    def evaluate_policy(self, state, action):
        v        = self.critic_net(state).squeeze()
        mean     = self.actor_net(state)
        dist     = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action)
        return v, log_prob


    def calculate_rewards_to_go(self, batch_rewards, batch_dones):
        rtgs = []

        discounted_reward = 0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)

        batch_rtgs = torch.tensor(rtgs, dtype=torch.float).to(self.device)
        return batch_rtgs


    def train_policy(self, memory):

        states      = torch.FloatTensor(np.asarray(memory.states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(memory.actions)).to(self.device)
        log_probs   = torch.FloatTensor(np.asarray(memory.log_probs)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(memory.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(memory.next_states)).to(self.device)
        dones       = torch.LongTensor(np.asarray(memory.dones)).to(self.device)

        # Reshape to batch_size x whatever
        #batch_size = len(states)
        #rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        #dones = dones.unsqueeze(0).reshape(batch_size, 1)

        logging.info(f"States {states.shape}")
        logging.info(f"actions {actions.shape}")
        logging.info(f"log_probs {log_probs.shape}")
        logging.info(f"rewards {rewards.shape}")
        logging.info(f"next_states {next_states.shape}")
        logging.info(f"dones {dones.shape}")


        # compute reward to go:
        rtgs = self.calculate_rewards_to_go(rewards, dones)
        #rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-7) # Normalizing the rewards, do I need this?
        logging.info(f"rtgs {rtgs.shape}")

        # calculate advantages
        v, _ = self.evaluate_policy(states, actions)
        logging.info(f"V {v.shape}")

        #advantages = rtgs.detach() - v.detach()
        advantages = rtgs - v.detach()
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)  # Normalize advantages,  do I need this?
        logging.info(f"advantages {advantages.shape}")


        for _ in range(self.k):
            v, curr_log_probs = self.evaluate_policy(states, actions)

            logging.info(f"new v {v.shape}")
            logging.info(f"new log probe v {curr_log_probs.shape}")

            # Calculate ratios
            #ratios = torch.exp(curr_log_probs - log_probs.detach())
            ratios = torch.exp(curr_log_probs - log_probs.squeeze())
            logging.info(f"ratios  {ratios.shape}")

            # Finding Surrogate Loss
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            logging.info(f"surr1  {surr1.shape}")
            logging.info(f"surr2  {surr2.shape}")


            # final loss of clipped objective PPO
            actor_loss  = (-torch.minimum(surr1, surr2)).mean()
            critic_loss = F.mse_loss(v, rtgs)

            self.actor_net.optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_net.optimiser.step()

            self.critic_net.optimiser.zero_grad()
            critic_loss.backward()
            self.critic_net.optimiser.step()

        memory.clear()
