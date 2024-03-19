import random
from collections import deque


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
