from __future__ import annotations
from collections import deque
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

    def __init__(self, max_capacity: int = int(1e6), dtype=np.float32):
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
        self.dtype = dtype
        self.full = False
        self.buffers = {}

    def add(self, **experience):
        """
        Adds an experience to the memory buffer. Each key-value pair in the experience dictionary is added
        to a corresponding buffer. If a buffer for a given key does not exist, it is created.

        If the memory buffer is full, this function will overwrite the oldest experience with the new one.

        Parameters
        ----------

        **experience : dict
            A dictionary where the key is the name of the experience and the value is the experience data.
            The data can be of any shape and will be converted to a numpy array if not already one. The
            experience data is expected to be numerical (i.e., integers or floats).

        Examples
        --------
        >>> memory.add(state=[1,2,3], action=1, reward=10, next_state=[2,3,4], done=False)
        """

        for key, value in experience.items():
            if key not in self.buffers:
                value = np.array(value, ndmin=1)  # Ensure value is at least 1D
                value_shape = np.shape(value)
                self.buffers[key] = np.empty((self.max_capacity, *value_shape), dtype=self.dtype)

            # Ensure value is at least 1D
            value = np.array(value, ndmin=1)
            np.copyto(self.buffers[key][self.head], value)

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
        indices = self._sample_indices(batch_size)
        sampled_experiences = {'indices': indices}
        for key in self.buffers:
            sampled_experiences[key] = self._sample_experience(key, indices)
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
        return np.random.choice(len(self), size=batch_size, replace=False)

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
        experiences = self.sample(len(self))
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
