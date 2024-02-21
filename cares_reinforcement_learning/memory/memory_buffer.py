import random
from collections import deque
import numpy as np


class MemoryBuffer:
    """ """

    def __init__(self, max_capacity=int(1e6)):
        self.buffer = deque([], maxlen=max_capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, state, action, reward, next_state, done, log_prob=0.0):
        """
        Add experiences to deque
        """
        experience = (state, action, reward, next_state, done, log_prob)
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample for policy
        """
        batch_size = min(batch_size, len(self.buffer))
        experience_batch = random.sample(self.buffer, batch_size)
        transposed_batch = zip(*experience_batch)
        self.sample_next(batch_size=batch_size)
        return transposed_batch

    def sample_next(self, batch_size):
        """
        For MBRL
        """
        batch_size = min(batch_size, len(self.buffer))
        max = len(self.buffer)
        idxs = np.random.randint(0, (max - 1), size=batch_size)
        # A list of tuples
        experience_batch = [
            self.buffer[i]
            + (
                self.buffer[i + 1][1],
                self.buffer[i + 1][2],
            )
            for i in idxs
        ]

        # experience_batch = random.sample(self.buffer, batch_size)
        transposed_batch = zip(*experience_batch)
        return transposed_batch

    def flush(self):
        """
        For PPO usage
        """
        states, actions, rewards, next_states, dones, log_probs = zip(
            *[(element[i] for i in range(len(element))) for element in self.buffer]
        )
        self.buffer.clear()
        return states, actions, rewards, next_states, dones, log_probs

    def get_statistics(self):
        """
        Compute the statisitics for world model normalization.
        :return:
        """

        states = [tuple[0] for tuple in self.buffer]
        next_states = [tuple[3] for tuple in self.buffer]
        states = np.array(states)
        next_states = np.array(next_states)
        delta = next_states - states

        statistics = {
            "ob_mean": np.mean(states, axis=0) + 0.0001,
            "ob_std": np.std(states, axis=0) + 0.0001,
            "delta_mean": np.mean(delta, axis=0) + 0.0001,
            "delta_std": np.std(delta, axis=0) + 0.0001,
        }

        return statistics
