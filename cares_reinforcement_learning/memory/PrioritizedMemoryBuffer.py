import numpy as np
from collections import deque
from cares_reinforcement_learning.memory import MemoryBuffer


class PrioritizedMemoryBuffer(MemoryBuffer):
    """
    This class represents a prioritized memory buffer used in Reinforcement Learning (RL).
    It overrides the _add_experience and _sample_experience methods of MemoryBuffer to add
    and sample experiences based on their priorities.

    Attributes
    ----------
    eps : float
        A small value to avoid zero priority.
    alpha : float
        A factor that determines how much prioritization is used.
    """

    def __init__(self, max_capacity=int(1e6), eps=1e-6, alpha=0.6):
        """
        The constructor for PrioritizedMemoryBuffer class.

        Parameters
        ----------
        max_capacity : int
            The maximum capacity of the buffer (default is 1,000,000).
        eps : float
            A small value to avoid zero priority (default is 1e-6).
        alpha : float
            A factor that determines how much prioritization is used (default is 0.6).
        """
        super().__init__(max_capacity)
        self.eps = eps
        self.alpha = alpha
        self.priorities = deque(maxlen=max_capacity)

    def _add_experience(self, key, value):
        """
        Adds an experience to a specific buffer and assigns maximum priority to it.

        Parameters
        ----------
        key : str
            The name of the buffer.
        value : object
            The experience to add.
        """
        super()._add_experience(key, value)
        max_priority = max(self.priorities) if self.priorities else 1.0
        self.priorities.append(max_priority)

    def _sample_experience(self, key, batch_size):
        """
        Samples a batch of experiences from a specific buffer based on their priorities.

        Parameters
        ----------
        key : str
            The name of the buffer.
        batch_size : int
            The size of the batch to sample.

        Returns
        -------
        list
            A list of experiences from a specific buffer.
        """
        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(len(self.buffers[key]), batch_size, p=probabilities)
        return [self.buffers[key][i] for i in indices]
