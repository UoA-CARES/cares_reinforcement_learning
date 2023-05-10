import numpy as np
from cares_reinforcement_learning.util import SumTree


class PrioritizedReplayBuffer():
    def __init__(self, state_dim, action_dim, max_capacity=int(1e6)):
        self.max_size = max_capacity
        self.ptr = 0
        self.size = 0
        self.device = DEVICE

        self.state = np.zeros((max_capacity, state_dim))
        self.action = np.zeros((max_capacity, action_dim))
        self.next_state = np.zeros((max_capacity, state_dim))
        self.reward = np.zeros((max_capacity, 1))
        self.done = np.zeros((max_capacity, 1))

        self.tree = SumTree(max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4

    def add(self, state, action, reward, next_state, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.done[self.ptr] = done

        self.tree.set(self.ptr, self.max_priority)
        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample(self, batch_size):
        ind = self.tree.sample(batch_size)

        weights = self.tree.levels[-1][ind] ** -self.beta
        weights /= weights.max()

        self.beta = min(self.beta + 2e-7, 1)

        return (
          torch.FloatTensor(self.state[ind]).to(self.device),
          torch.FloatTensor(self.action[ind]).to(self.device),
          torch.FloatTensor(self.next_state[ind]).to(self.device),
          torch.FloatTensor(self.reward[ind]).to(self.device),
          torch.FloatTensor(self.done[ind]).to(self.device),
          ind,
          torch.FloatTensor(weights).to(self.device).reshape(-1, 1)
        )

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)