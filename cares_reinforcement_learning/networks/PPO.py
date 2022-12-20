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
        # print(f"Batch Size: {len(states)}")
        batch_size = len(states)
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.FloatTensor(np.asarray(actions))
        rewards = torch.FloatTensor(np.asarray(rewards))
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.LongTensor(np.asarray(dones))
        # old_log_probs = torch.FloatTensor(old_log_probs)
        #
        # print(f"{states=}")
        # print(f"{actions=}")
        # print(f"{rewards=}")
        # print(f"{next_states=}")
        # print(f"{dones=}")
        # print(f"{old_log_probs=}")

        rewards_to_go, rewards = self.compute_rewards_to_go(rewards)
        rewards_to_go = torch.FloatTensor(rewards_to_go)
        # Calculate the Advantages using Generalise Advantage Estimation
        advantages = []

        adv_batch = list(zip(states, rewards, next_states, dones))

        for state, reward, next_state, done in reversed(adv_batch):
            # print(state, reward, next_state, done)

            v_t = self.critic(state).squeeze(0)
            v_t_plus_1 = self.critic(next_state).squeeze(0)

            # print(reward, self.gamma, v_t_plus_1, v_t)
            delta_t = reward + self.gamma * v_t_plus_1 * done - v_t
            adv_t_plus_1 = advantages[-1] if advantages else 0

            # print(f"{delta_t=}")
            # print(f"{adv_t_plus_1=}")
            advantage = delta_t + self.lamda * self.gamma * adv_t_plus_1
            advantages.append(advantage)

        advantages = list(reversed(advantages))
        advantages = torch.tensor(advantages)

        # Calculating the policy targets
        bound = torch.where(advantages >= 0, advantages * (1 + self.epsilon), advantages * (1 - self.epsilon))

        mean, log_std = self.actor(states)
        # print(f"{mean=} {log_std=}")

        # mean and log_std have grad

        std_dev = log_std.exp()

        dists = [Normal(mean, std_dev) for mean, std_dev in zip(mean, std_dev)]
        # print(f"{dists=}")
        new_log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]

        # print(np.shape(new_log_probs))
        # print(f"{new_log_probs=}")

        # New Log Probs has grad

        # new_log_probs = torch.tensor(new_log_probs)

        old_log_probs = torch.stack(list(old_log_probs))
        new_log_probs = torch.stack(new_log_probs)
        # print(f"{new_log_probs=}")
        # print(f"{old_log_probs=}")
        ratio = (new_log_probs - old_log_probs).exp()

        policy_target = torch.min(ratio * advantages, bound)

        # Ratio no grad
        # Advantage no grad
        # Bound no grad

        # print(f"{policy_target=}")
        # print(f"{ratio=}")
        # print(f"{advantages=}")
        # print(f"{bound=}")
        # Calculating Policy Loss
        actor_loss = -policy_target.mean()
        # print(f"{actor_loss}")

        self.actor.optimiser.zero_grad()
        actor_loss.backward()
        self.actor.optimiser.step()

        # Updating Critic Network
        V = torch.reshape(self.critic(states), (batch_size,))
        # print(f"{V=}")
        # print(f"{rewards_to_go=}")
        critic_loss = self.critic.loss(V, rewards_to_go)


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
        ordinary_reward = []

        for episode_rewards in reversed(rewards):

            discounted_reward = 0

            for reward in reversed(episode_rewards):
                discounted_reward = reward + discounted_reward * self.gamma
                reward_to_gos.insert(0, discounted_reward)
                ordinary_reward.insert(0, reward)

        return reward_to_gos, ordinary_reward

