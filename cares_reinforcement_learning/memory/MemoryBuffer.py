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

    def __init__(self, max_capacity: int | None = int(1e6)):
        """
        The constructor for MemoryBuffer class.

        Parameters
        ----------
        max_capacity : int
            The maximum capacity of the buffer (default is 1,000,000).
        """
        self.max_capacity = max_capacity
        self.buffers = {}

    def add(self, **experience):
        """
        Adds an experience to the buffer.

        Parameters
        ----------
        experience : dict
            The dictionary of experiences. Keys are the names of the buffers,
            and values are the experiences.
        """
        for key, value in experience.items():
            if key not in self.buffers:
                self.buffers[key] = deque(maxlen=self.max_capacity)
            self._add_experience(key, value)

    def _add_experience(self, key, value):
        """
        Adds an experience to a specific buffer.

        Parameters
        ----------
        key : str
            The name of the buffer.
        value : object
            The experience to add.
        """
        self.buffers[key].append(value)

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
        buffer_length = len(next(iter(self.buffers.values())))
        return np.random.choice(buffer_length, size=batch_size, replace=False)

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
        for buffer in self.buffers.values():
            buffer.clear()

    def flush(self):
        """
        Flushes all the buffers and returns all experiences.

        Returns
        -------
        dict
            A dictionary of all experiences. Keys are the names of the buffers,
            and values are the lists of experiences.
        """
        experiences = {key: list(buffer) for key, buffer in self.buffers.items()}
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
        return len(next(iter(self.buffers.values())))
