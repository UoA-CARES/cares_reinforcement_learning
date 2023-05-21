from cares_reinforcement_learning.memory import MemoryBuffer
from collections import deque
import numpy as np


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

    def __init__(self, max_capacity: int | None = int(1e6), eps=1e-6, alpha=0.6):
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
        self.buffers['priorities'] = deque(maxlen=max_capacity)

    def add(self, **experience):
        """
        Adds an experience to the buffer.

        Parameters
        ----------
        experience : dict
            The dictionary of experiences. Keys are the names of the buffers,
            and values are the experiences.
        """
        super().add(**experience)
        max_priority = max(self.buffers['priorities']) if self.buffers['priorities'] else 1.0
        self.buffers['priorities'].append(max_priority)

    def _sample_indices(self, batch_size):
        """
        Samples a batch of indices for experiences.

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample.

        Returns
        -------
        list
            A list of indices.
        """
        priorities = np.array(self.buffers['priorities'], dtype=np.float32).flatten()
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        return np.random.choice(len(self), batch_size, p=probabilities)

    def update_priorities(self, indices, errors, offset=0.1):
        """
        Updates the priorities of the experiences based on the TD errors from learning update.

        Parameters
        ----------
        indices : list
            The indices of the experiences.
        errors : list
            The list of errors (TD errors).
        offset : float
            A small positive constant to avoid zero priority.

        """
        for idx, error in zip(indices, errors):
            self.buffers['priorities'][idx] = abs(error) + self.eps + offset
