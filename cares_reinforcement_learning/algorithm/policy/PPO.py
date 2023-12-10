"""
Original Paper:
                https://arxiv.org/abs/1707.06347
Good Explanation:
                https://www.youtube.com/watch?v=5P7I-xPq8u8
Code based on:
                https://github.com/ericyangyu/PPO-for-Beginners
                https://github.com/nikhilbarhate99/PPO-PyTorch
"""
# Deliberate naming convention to match papers and other implementations
# pylint: disable-next=invalid-name

import os
import logging
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal


class PPO:
    def __init__(
        self,
        actor_network,
        critic_network,
        gamma,
        action_num,
        actor_lr,
        critic_lr,
        device,
    ):
        self.type = "policy"
        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.gamma = gamma
        self.action_num = action_num
        self.device = device

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        self.k = 10
        self.eps_clip = 0.2
        self.cov_var = torch.full(size=(action_num,), fill_value=0.5).to(self.device)
        self.cov_mat = torch.diag(self.cov_var)

    def select_action_from_policy(self, state):
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)

            mean = self.actor_net(state_tensor)
            dist = MultivariateNormal(mean, self.cov_mat)

            # Sample an action from the distribution and get its log prob
            action = dist.sample()
            log_prob = dist.log_prob(action)

            action = action.cpu().data.numpy().flatten()
            log_prob = (
                log_prob.cpu().data.numpy().flatten()
            )  # just to have this as numpy array
        self.actor_net.train()
        return action, log_prob

    def evaluate_policy(self, state, action):
        v = self.critic_net(state).squeeze()  # shape 5000
        mean = self.actor_net(state)  # shape, 5000, 1
        dist = MultivariateNormal(mean, self.cov_mat)
        log_prob = dist.log_prob(action)  # shape, 5000
        return v, log_prob

    def calculate_rewards_to_go(self, batch_rewards, batch_dones):
        rtgs = []
        discounted_reward = 0
        for reward, done in zip(reversed(batch_rewards), reversed(batch_dones)):
            discounted_reward = reward + self.gamma * (1 - done) * discounted_reward
            rtgs.insert(0, discounted_reward)
        batch_rtgs = torch.tensor(rtgs, dtype=torch.float).to(self.device)  # shape 5000
        return batch_rtgs

    def train_policy(self, experience):
        info = {}

        states, actions, rewards, next_states, dones, log_probs = experience

        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        log_probs = torch.FloatTensor(np.asarray(log_probs)).to(self.device)

        log_probs = log_probs.squeeze()  # torch.Size([5000])

        # compute reward to go:
        rtgs = self.calculate_rewards_to_go(rewards, dones)  # torch.Size([5000])
        # rtgs = (rtgs - rtgs.mean()) / (rtgs.std() + 1e-7)

        # calculate advantages
        v, _ = self.evaluate_policy(states, actions)  # torch.Size([5000])

        advantages = rtgs.detach() - v.detach()  # torch.Size([5000])
        advantages = (advantages - advantages.mean()) / (
            advantages.std() + 1e-10
        )  # Normalize advantages

        td_errors = torch.abs(advantages)
        td_errors = td_errors.data.cpu().numpy()

        for _ in range(self.k):
            v, curr_log_probs = self.evaluate_policy(states, actions)

            # Calculate ratios
            ratios = torch.exp(
                curr_log_probs - log_probs.detach()
            )  # torch.Size([5000])

            # Finding Surrogate Loss
            surr1 = ratios * advantages  # torch.Size([5000])
            surr2 = (
                torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            )  # torch.Size([5000])

            # final loss of clipped objective PPO
            actor_loss = (-torch.minimum(surr1, surr2)).mean()
            critic_loss = F.mse_loss(v, rtgs)

            self.actor_net_optimiser.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_net_optimiser.step()

            self.critic_net_optimiser.zero_grad()
            critic_loss.backward()
            self.critic_net_optimiser.step()

        info["td_error"] = td_errors
        info["actor_loss"] = actor_loss
        info["critic_loss"] = critic_loss

        return info

    def save_models(self, filename, filepath="models"):
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
