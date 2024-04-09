import numpy as np
import torch

from cares_reinforcement_learning.util.sum_tree import SumTree


class ActorPrioritizedReplayBuffer:
    def __init__(self, state_dim, action_dim, max_size=int(1e6), device=None):
        self.device = device

        self.max_size = max_size
        self.ptr = 0
        self.size = 0

        self.state = np.zeros((max_size, state_dim))
        self.action = np.zeros((max_size, action_dim))
        self.next_state = np.zeros((max_size, state_dim))
        self.reward = np.zeros((max_size, 1))
        self.not_done = np.zeros((max_size, 1))

        self.critic_tree = SumTree(max_size)

        self.max_priority_critic = 1.0

        self.new_tree = SumTree(max_size)

        self.beta_critic = 0.4

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1.0 - done

        self.critic_tree.set(self.ptr, self.max_priority_critic)

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_uniform(self, batch_size):
        ind = np.random.randint(self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            None,
        )

    def sample_critic(self, batch_size):
        ind = self.critic_tree.sample(batch_size)

        weights = self.critic_tree.levels[-1][ind] ** -self.beta_critic
        weights /= weights.max()

        self.beta_critic = min(self.beta_critic + 2e-7, 1)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            torch.FloatTensor(weights).to(self.device).reshape(-1, 1),
        )

    def sample_actor(self, batch_size, t):
        top_value = self.critic_tree.levels[0][0]

        reversed_priorities = top_value / (
            self.critic_tree.levels[-1][: self.ptr] + 1e-6
        )

        if self.ptr != 0:
            self.new_tree.batch_set_v2(np.arange(self.ptr), reversed_priorities, t)

        ind = self.new_tree.sample(batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.FloatTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device),
            ind,
            torch.FloatTensor(reversed_priorities[ind]).to(self.device).reshape(-1, 1),
        )

    def update_priority_critic(self, ind, priority):
        self.max_priority_critic = max(priority.max(), self.max_priority_critic)
        self.critic_tree.batch_set(ind, priority)
