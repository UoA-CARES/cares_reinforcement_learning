import numpy as np
import torch

from cares_reinforcement_learning.util.sum_tree import SumTree


class PrioritizedReplayBuffer:

    def __init__(
        self,
        max_capacity=int(1e6),
        **priority_params,
    ):
        self.priority_params = priority_params

        self.max_capacity = max_capacity

        self.ptr = 0
        self.size = 0

        self.memory_buffers = []

        self.tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4

    def __len__(self):
        return self.size

    def add(self, *experience):
        # Dynamically create the full memory size on first experience
        for index, exp in enumerate(experience):
            if index >= len(self.memory_buffers):
                exp_size = 1 if isinstance(exp, (int, float)) else exp.shape[0]
                self.memory_buffers.append(np.zeros((self.max_capacity, exp_size)))

            self.memory_buffers[index][self.ptr] = exp

        self.tree.set(self.ptr, self.max_priority)

        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample(self, batch_size):
        # if batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.size)
        indices = self.tree.sample(batch_size)

        weights = self.tree.levels[-1][indices] ** -self.beta
        weights /= weights.max()

        # Prevents priorities from being zero
        self.beta = min(self.beta + 2e-7, 1)

        experiences = []
        for buffer in self.memory_buffers:
            experiences.append(buffer[indices])

        return (
            *experiences,
            indices,
            weights,
        )

    def update_priority(self, info):
        ind = info["indices"]
        priorities = (
            info["priorities"]
            if "priorities" in info
            else torch.tensor([1.0] * len(info["indices"]))
        )

        self.max_priority = max(priorities.max(), self.max_priority)
        self.tree.batch_set(ind, priorities)

    def flush(self):
        experiences = []
        for buffer in self.memory_buffers:
            experiences.append(buffer[0 : self.size])
        self.clear()
        return experiences

    def clear(self):
        self.ptr = 0
        self.size = 0
        self.memory_buffers = []

        self.tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4

    # def get_statistics(self):
    #     """
    #     Compute the statisitics for world model state normalization.
    #     state, action, reward, next_state, done

    #     :return: statistic tuple of the collected transitions.
    #     """
    #     states = [trans[0] for trans in self.buffer]
    #     next_states = [trans[3] for trans in self.buffer]

    #     states = np.array(states)
    #     next_states = np.array(next_states)
    #     delta = next_states - states

    #     # Add a small number to avoid zeros.
    #     observation_mean = np.mean(states, axis=0) + 0.00001
    #     observation_std = np.std(states, axis=0) + 0.00001
    #     delta_mean = np.mean(delta, axis=0) + 0.00001
    #     delta_std = np.std(delta, axis=0) + 0.00001

    #     statistics = {
    #         "observation_mean": observation_mean,
    #         "observation_std": observation_std,
    #         "delta_mean": delta_mean,
    #         "delta_std": delta_std,
    #     }
    #     return statistics
