import random
from collections import deque
import numpy as np


class MemoryBuffer:
    """
    A memory buffer to temporarily store transitions. It is like a transition
    pool. Only off-policy algorithms needs random sampling this buffer.

    """

    def __init__(self, max_capacity=int(1e6)):
        self.buffer = deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, *experience):
        """
        Add new transitions into the buffer. Be aware of the sequence. Should
        be consistent with how it is sampled.

        Example:

        memory.add(state, action, reward, next_state, done)

        :param experience: a tuple of a transition.
        """
        self.buffer.append(experience)

    def sample(self, batch_size):
        """
        Sample for training the agent.

        :param batch_size: It is pre-set by training configuration file.
        """
        batch_size = min(batch_size, len(self.buffer))
        experience_batch = random.sample(self.buffer, batch_size)
        transposed_batch = zip(*experience_batch)
        return transposed_batch

    def sample_next(self, batch_size):
        """
        For MBRL to train to predict rewards. The right next transition is
        chosen, WHEN THE BUFFER IS NOT SHUFFLED.

        """
        batch_size = min(batch_size, len(self.buffer) - 1)
        max_length = len(self.buffer)
        idxs = np.random.randint(0, (max_length - 1), size=batch_size)
        # A list of tuples
        experience_batch = [
            self.buffer[i]
            + (
                self.buffer[i + 1][1],
                self.buffer[i + 1][2],
            )
            for i in idxs
        ]
        transposed_batch = zip(*experience_batch)
        return transposed_batch

    def flush(self):
        """
        For on-policy PPO. Output all collected trajectories.
        And clear the buffer.

        :return:
        """
        experience = list(zip(*self.buffer))
        self.buffer.clear()
        return experience

    def clear(self):
        """
        Clear the buffer.
        """
        self.buffer.clear()

    def get_statistics(self):
        """
        Compute the statisitics for world model state normalization.
        state, action, reward, next_state, done

        :return: statistic tuple of the collected transitions.
        """
        states = [trans[0] for trans in self.buffer]
        next_states = [trans[3] for trans in self.buffer]

        states = np.array(states)
        next_states = np.array(next_states)
        delta = next_states - states

        ob_mean = np.mean(states, axis=0) + 0.00001
        ob_std = np.std(states, axis=0) + 0.00001
        delta_mean = np.mean(delta, axis=0) + 0.00001
        delta_std = np.std(delta, axis=0) + 0.00001

        statistics = {
            "ob_mean": ob_mean,
            "ob_std": ob_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        return statistics
