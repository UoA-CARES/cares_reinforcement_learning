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

    def learn(self, states, actions, rewards, next_states, dones, old_log_probs, num_updates):

        """
        Args:
            states: an array of observations
            actions: an array of actions taken
            rewards: an array of rewards
            next_states: array of observations of the next state
            dones: array of dones
            old_log_probs: an array of log probabilities for the action taken (old)
            num_updates: an integer determining how many times the policy updates
        """

        rtg = self.compute_rewards_to_go(rewards)

        advantages = self.compute_advantages(states, rewards, next_states, dones)

        for _ in range(0, num_updates):
            # Calculate Bound
            clipped_advantages = torch.where(advantages >= 0, advantages * (1 + self.epsilon),
                                             advantages * (1 - self.epsilon))

            # Computing the log probabilities of actions taken using current policy
            mean, log_std = self.actor(states)

            std_dev = log_std.exp()

            dists = [Normal(mean, std_dev) for mean, std_dev in zip(mean, std_dev)]

            new_log_probs = [dist.log_prob(action) for dist, action in zip(dists, actions)]

            ratio = (new_log_probs - old_log_probs).exp()

            policy_target = torch.min(ratio * advantages, clipped_advantages)

            actor_loss = -policy_target.mean()

            self.actor.optimiser.zero_grad()
            actor_loss.backward()
            self.actor.optimiser.step()

            v = self.critic(states)

            critic_loss = self.critic.loss(v, rtg)

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

    def compute_advantages(self, states, reward_to_gos, next_states, dones):
        """
        Use Generalised Advantage Estimation to compute advantages
        """

        advantages = []

        batch = list(zip(states, reward_to_gos, next_states, dones))

        for state, rtg, next_state, done in reversed(batch):
            v_t = self.critic(state)
            v_t_plus_one = self.critic(next_state)

            delta_t = rtg + self.gamma * v_t_plus_one * done - v_t
            adv_t_plus_one = advantages[-1] if advantages else 0

            advantage = delta_t + self.lamda * self.gamma * adv_t_plus_one
            advantages.insert(0, advantage)

        return advantages


if torch.cuda.is_available():
    DEVICE = torch.device('cuda')
    print("Working with GPU")
else:
    DEVICE = torch.device('cpu')
    print("Working with CPU")

BUFFER_CAPACITY = 10_000

GAMMA = 0.995
TAU = 0.005
LAMDA = 0.001

ACTOR_LR = 1e-4
CRITIC_LR = 1e-3

EPISODE_NUM = 100
BATCH_SIZE = 64

env = gym.make('Pendulum-v1', g=9.81)


def main():

    observation_size = env.observation_space.shape[0]
    action_num = env.action_space.shape[0]

    max_actions = env.action_space.high
    min_actions = env.action_space.low

    memory = MemoryBuffer(BUFFER_CAPACITY)

    actor = Actor(observation_size, action_num, ACTOR_LR, max_actions)
    critic_one = Critic(observation_size, action_num, CRITIC_LR)
    critic_two = Critic(observation_size, action_num, CRITIC_LR)

    ppo = PPO(
        actor=actor,
        critic=critic_one,
        gamma=GAMMA,
        lamda=0.001
    )

    print(f"Training Beginning")
    train(td3)


def train(td3):
    historical_reward = []

    for episode in range(0, EPISODE_NUM):

        state, _ = env.reset()
        episode_reward = 0

        while True:

            # Select an Action
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state)
                state_tensor = state_tensor.unsqueeze(0)
                state_tensor = state_tensor.to(DEVICE)
                action = td3.forward(state_tensor)
                action = action.cpu().data.numpy()

            action = action[0]

            next_state, reward, terminated, truncated, _ = env.step(action)

            memory.add(state, action, reward, next_state, terminated)

            experiences = memory.sample(BATCH_SIZE)

            for _ in range(0, 10):
                td3.learn(experiences)

            state = next_state
            episode_reward += reward

            if terminated or truncated:
                break

        historical_reward.append(episode_reward)
        print(f"Episode #{episode} Reward {episode_reward}")


if __name__ == '__main__':
    main()