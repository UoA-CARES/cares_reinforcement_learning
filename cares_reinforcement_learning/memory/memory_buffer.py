import random
from collections import deque
import numpy as np
import torch


class MemoryBuffer:
    """Buffer to store environment transitions."""

    def __init__(self, state_dim, action_dim, capacity=1e6):
        self.statistics = None
        self.obs_shape = state_dim
        self.action_shape = action_dim
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.obses = np.empty((capacity, *state_dim), dtype=np.float32)
        self.next_obses = np.empty((capacity, *state_dim), dtype=np.float32)
        self.actions = np.empty((capacity, *action_dim), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.dones = np.empty((capacity, 1), dtype=np.float32)

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(self, obs, action, reward, next_obs, done):
        """
        Add new transition in the buffer.
        :param obs:
        :param action:
        :param reward:
        :param next_obs:
        :param done:
        """
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.dones[self.idx], done)
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_next(self, batch_size):
        """
        Randomly Sample transitions from stored data
        :param batch_size:
        :return:
        """
        idxs = np.random.randint(
            0, (self.capacity - 1) if self.full else (self.idx - 1), size=batch_size
        )
        obses = self.obses[idxs]
        actions = self.actions[idxs]
        rewards = self.rewards[idxs]
        next_actions = self.actions[idxs + 1]
        next_rewards = self.rewards[idxs + 1]
        next_obses = self.next_obses[idxs]
        not_dones = self.dones[idxs]
        # experience["state"],
        # experience["action"],
        # experience["reward"],
        # experience["next_state"],
        # experience["done"],
        # experience["next_action"],
        # experience["next_reward"],
        return (
            obses,
            actions,
            rewards,
            next_obses,
            not_dones,
            next_actions,
            next_rewards,
        )

    def get_statistics(self):
        """
        Compute the statisitics for world model normalization.
        :return:
        """

        states = np.array(self.obses)
        next_states = np.array(self.next_obses)
        delta = next_states - states

        statistics = {
            "ob_mean": np.mean(states, axis=0) + 0.0001,
            "ob_std": np.std(states, axis=0) + 0.0001,
            "delta_mean": np.mean(delta, axis=0) + 0.0001,
            "delta_std": np.std(delta, axis=0) + 0.0001,
        }
        return statistics
