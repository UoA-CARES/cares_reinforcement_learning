import torch 
import numpy as np 
from .SumTree import SumTree

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class PrioritizedReplayBuffer():
    def __init__(self, max_size=int(1e6)):  # max_capacity
        self.max_size = max_size

        self.ptr = 0
        self.size = 0

        self.state = []
        self.action = []
        self.next_state = []
        self.reward = []
        self.done = []

        self.tree = SumTree(max_size)
        self.max_priority = 1.0
        self.beta = 0.4
        self.epsilon_d = 1e-6

        self.device = DEVICE

    def add(self, state, action, reward, next_state, done):
        self.state.append(state)
        self.action.append(action)
        self.next_state.append(next_state)
        self.reward.append(reward)
        self.done.append(done)

        import time
        time_start = time.time()
        self.tree.set(self.ptr, self.max_priority)
        time_end = time.time()
        # print(f"Time taken to set priority: {time_end - time_start}")
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = self.tree.sample(batch_size)

        weights = self.tree.levels[-1][ind] ** -self.beta
        weights /= weights.max()

        self.beta = min(self.beta + 2e-7, 1)

        return (
            torch.FloatTensor(np.array(self.state)[ind]).to(self.device),
            torch.FloatTensor(np.array(self.action)[ind]).to(self.device),
            torch.FloatTensor(np.array(self.reward)[ind]).to(self.device),
            torch.FloatTensor(np.array(self.next_state)[ind]).to(self.device),
            torch.FloatTensor(np.array(self.done)[ind]).to(self.device),
            ind,
            torch.FloatTensor(weights).to(self.device).reshape(-1, 1)
        )

    def update_priority(self, ind, priority):
        self.max_priority = max(priority.max(), self.max_priority)
        self.tree.batch_set(ind, priority)