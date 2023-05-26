from __future__ import annotations
from collections import deque
from cares_reinforcement_learning.memory.augments import *
import numpy as np


class MemoryBuffer:
    """
    This class represents a base memory buffer used in Reinforcement Learning (RL).
    The buffer is used to store experiences of the interactions of an agent with its environment.

    The buffer has a maximum capacity and uses a deque to store the experiences,
    because it is efficient for appending and popping on both ends especially when
    the maximum length is reached.

    Attributes
    ----------
    max_capacity : int
        The maximum capacity of the buffer.
    buffers : dict
        A dictionary to hold different buffers for easy management.
    """

    def __init__(self, max_capacity: int = int(1e6),
                 eps=1e-6, alpha=0.6, augment=std, **params):
        """
        The constructor for MemoryBuffer class.

        Parameters
        ----------
        max_capacity : int
            The maximum capacity of the buffer (default is 1,000,000).
        dtype : np value type
            Sets the value type data is stored as
        """
        self.max_capacity = max_capacity
        self.head = 0
        self.full = False

        self.buffers = {'priorities': deque(maxlen=max_capacity)}

        self.augment = augment
        self.params = {"eps": eps, "alpha": alpha}
        for key, value in params.items():
            self.params[key] = value

    def add(self, **experience):
        """
        Adds an experience to the memory buffer. Each key-value pair in the experience dictionary is added
        to a corresponding buffer. If a buffer for a given key does not exist, it is created.

        If the memory buffer is full, this function will overwrite the oldest experience with the new one.

        Parameters
        ----------

        **experience : dict
            A dictionary where the key is the name of the experience and the value is the experience data.
            The data can be of any type.

        Examples
        --------
        >>> memory.add([1,2,3], [2,3,4], action=[1.5, 2.1],
        >>>            reward=-0.32, done=False)
        """
        for key, value in experience.items():
            if key not in self.buffers:
                self.buffers[key] = [None] * self.max_capacity

            self.buffers[key][self.head] = value

        if self.buffers.get("priorities") is None:
            self.buffers["priorities"] = deque(maxlen=self.max_capacity)
        max_priority = max(self.buffers['priorities']) if self.buffers['priorities'] else 1.0
        self.buffers['priorities'].append(max_priority)

        self.head = (self.head + 1) % self.max_capacity
        self.full = self.full or self.head == 0

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffers.

        Parameters
        ----------
        batch_size : int
            The size of the batch to sample.

        Returns
        -------
        dict
            A dictionary of batched experiences. Keys are the names of the buffers,
            and values are the lists of experiences.
        """
        batch_size = batch_size if len(self) > batch_size else len(self)
        indices = self._sample_indices(batch_size)
        sampled_experiences = {'indices': indices}
        for key in self.buffers:
            sampled_experiences[key] = [self.buffers[key][i] for i in indices]
        return sampled_experiences

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
        probabilities = priorities ** self.params["alpha"]
        probabilities /= probabilities.sum()
        return np.random.choice(len(self), size=batch_size, p=probabilities, replace=False)

    def _sample_experience(self, key, indices):
        """
        Samples a batch of experiences from a specific buffer using given indices.

        Parameters
        ----------
        key : str
            The name of the buffer.
        indices : list
            The indices to use for sampling.

        Returns
        -------
        list
            A list of experiences from a specific buffer.
        """
        return [self.buffers[key][i] for i in indices]

    def update_priorities(self, indices, info):
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
        errors = self.augment(indices, info, self.params)
        for idx, error in zip(indices, errors):
            self.buffers['priorities'][idx] = abs(error) + self.params["eps"]

    def clear(self):
        """
        Clears all the buffers.
        """
        self.head = 0
        self.full = False
        self.buffers.clear()

    def flush(self):
        """
        Flushes all the buffers and returns all experiences.

        Returns
        -------
        dict
            A dictionary of all experiences. Keys are the names of the buffers,
            and values are the lists of experiences.
        """
        experiences = {'indices': range(len(self))}
        for key in self.buffers:
            experiences[key] = [self.buffers[key][i] for i in experiences['indices']]

        self.clear()
        return experiences

    def __len__(self):
        """
        Returns the number of experiences in the buffer.

        Returns
        -------
        int
            The number of experiences in any buffer. Assumes all buffers have the same length.
        """
        return self.max_capacity if self.full else self.head
