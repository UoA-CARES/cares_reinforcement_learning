import copy

import torch
from torch.distributions import Normal
import numpy as np


class PPO:
    def __init__(self,
                 actor,
                 critic,
                 gamma,
                 lamda,
                 epsilon):

        self.actor = actor
        self.critic = critic

        self.gamma = gamma
        self.lamda = lamda
        self.epsilon = epsilon

    def forward(self, state):
        return self.actor(state)

    def learn(self, states, actions, rewards, next_states, dones, old_log_probs):
        """

        Args:
            states: an array of observations
            actions: an array of actions taken
            rewards: an array of rewards
            next_states: array of observations of the next state
            dones: array of dones
            old_log_probs: an array of log probabilities for the action taken (old)

        Returns:

        """

        rewards_to_go = self.compute_rewards_to_go(rewards)

        # Calculate the Advantages using Generalise Advantage Estimation
        advantages = []

        adv_batch = list(zip(states, rewards, next_states, dones))

        for state, reward, next_state, done in reversed(adv_batch):
            delta_t = reward + self.gamma * self.critic(next_state).detach() * done - self.critic(state).detach()
            adv_t_plus_1 = adv_batch[-1] if adv_batch else 0

            advantage = delta_t + self.lamda * self.gamma * adv_t_plus_1
            advantages.append(advantage)

        advantages = list(reversed(advantages))
        advantages = torch.tensor(advantages)

        # Calculating the policy targets
        bound = torch.where(advantages >= 0, advantages * (1 + self.epsilon), advantages * (1 - self.epsilon))

        mean, std_dev = self.actor(states)

        dist: Normal = Normal(mean, std_dev)
        new_log_probs = dist.log_prob(actions)

        ratio = (new_log_probs - old_log_probs).exp()

        policy_target = torch.min(ratio * advantages, bound)

        # Calculating Policy Loss
        actor_loss = -policy_target.mean()

        self.actor.optimiser.zero_grad()
        actor_loss.backward()
        self.actor.optimiser.step()

        # Updating Critic Network
        critic_loss = self.critic.loss(self.critic(states).detach(), rewards_to_go)

        self.critic.optimiser.zero_grad()
        critic_loss.backward()
        self.critic.optimiser.step()

    def compute_rewards_to_go(self, rewards):
        """
        Here we compute the rewards-to-go for all the observations
        Args:
            rewards: array of rewards
            episode_nums: array of episode numbers the rewards come from

        Returns:
            an array of discounted rewards
        """

        reward_to_gos = []

        for episode_rewards in reversed(rewards):

            discounted_reward = 0

            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                reward_to_gos.insert(0, discounted_reward)

        return reward_to_gos

