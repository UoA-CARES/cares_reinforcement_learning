from __future__ import annotations
from collections import deque
from cares_reinforcement_learning.memory import MemoryBuffer
import numpy as np
from collections.abc import Callable
from cares_reinforcement_learning.memory.per_augments import td_error


class PrioritizedMemoryBuffer(MemoryBuffer):
    """
    This class represents a prioritized memory buffer used in Reinforcement Learning (RL).
    The buffer is an extension of the base MemoryBuffer class and is used to store experiences
    of the interactions of an agent with its environment.

    The buffer assigns priorities to experiences based on their errors, and the sampling of
    experiences for training is biased towards higher-priority experiences.

    Attributes
    ----------
    max_capacity : int
        The maximum capacity of the buffer.
    eps : float, optional
        A small constant added to the priorities to ensure non-zero probabilities during sampling.
    alpha : float, optional
        The exponent used to transform priorities into probabilities during sampling.
    augment : Callable[[dict], list], optional
        A function used to augment errors before updating priorities.
    buffers : dict
        A dictionary to hold different buffers for easy management.
    """

    def __init__(self, max_capacity: int | None = int(1e6),
                 eps=1e-6, alpha=0.6, augment=td_error):
        """
        The constructor for PrioritizedMemoryBuffer class.

        Parameters
        ----------
        max_capacity : int, optional
            The maximum capacity of the buffer (default is 1,000,000).
        eps : float, optional
            A small constant added to the priorities to ensure non-zero probabilities during sampling.
        alpha : float, optional
            The exponent used to transform priorities into probabilities during sampling.
        augment
            A function used to augment errors before updating priorities.
        """
        super().__init__(max_capacity)
        self.eps = eps
        self.alpha = alpha
        self.augment = augment
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

    def update_priorities(self, indices, info, offset=0.1):
        """
        Updates the priorities of experiences at given indices.

        Parameters
        ----------
        indices : list
            The indices of experiences to update.
        info : dict
            Info dict returned from model training.
        offset : float, optional
            A small constant added to the absolute errors for updating priorities (default is 0.1).
        """
        errors = self.augment(info)
        for idx, error in zip(indices, errors):
            self.buffers['priorities'][idx] = abs(error) + self.eps + offset
