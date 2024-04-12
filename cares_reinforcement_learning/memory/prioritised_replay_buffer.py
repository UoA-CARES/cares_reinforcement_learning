import numpy as np
import random
from cares_reinforcement_learning.util.sum_tree import SumTree


class PrioritizedReplayBuffer:
    """
    A prioritized replay buffer implementation for reinforcement learning.

    This buffer stores experiences and samples them based on their priorities.
    Experiences can be added to the buffer and sampled with different probabilities
    based on their priorities. The buffer also supports updating the priorities
    of the stored experiences.

    Args:
        max_capacity (int): The maximum capacity of the replay buffer.
        **priority_params: Additional parameters for priority calculation.

    Attributes:
        priority_params (dict): Additional parameters for priority calculation.
        max_capacity (int): The maximum capacity of the replay buffer.
        ptr (int): The current position in the buffer.
        size (int): The current size of the buffer.
        memory_buffers (list): List of memory buffers for each experience type.
        tree (SumTree): The sum tree data structure for priority calculation.
        max_priority (float): The maximum priority value in the buffer.
        beta (float): The beta value for importance weight calculation.

    Methods:
        __len__(): Returns the current size of the buffer.
        add(*experience): Adds a single experience to the prioritized replay buffer.
        sample(batch_size): Samples experiences from the prioritized replay buffer.
        update_priority(info): Update the priorities of the replay buffer based on the given information.
        flush(): Flushes the memory buffers and returns the experiences in order.
        clear(): Clears the prioritized replay buffer.
    """

    def __init__(
        self,
        max_capacity=int(1e6),
        **priority_params,
    ):
        self.priority_params = priority_params

        self.max_capacity = max_capacity

        self.ptr = 0
        self.size = 0

        self.memory_buffers = []

        self.tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4

    def __len__(self):
        return self.size

    def add(self, *experience):
        """
        Adds a single experience to the prioritized replay buffer.

        Data is expected to be stored in the order: state, action, reward, next_state, done, ...

        Args:
            *experience: Variable number of experiences to be added. Each experience can be a single value or an array-like object.
        Returns:
            None
        """
        # Iteratre over the list of experiences (state, action, reward, next_state, done, ...) and add them to the buffer
        for index, exp in enumerate(experience):
            # Dynamically create the full memory size on first experience
            if index >= len(self.memory_buffers):
                # NOTE: This is a list of numpy arrays in order to use index extraction in sample O(1)
                memory = np.array([None] * self.max_capacity)
                self.memory_buffers.append(memory)

            # This adds to the latest position in the buffer
            self.memory_buffers[index][self.ptr] = exp

        self.tree.set(self.ptr, self.max_priority)

        self.ptr = (self.ptr + 1) % self.max_capacity
        self.size = min(self.size + 1, self.max_capacity)

    def sample(self, batch_size):
        """
        Samples experiences from the prioritized replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Tuple: A tuple containing the sampled experiences, indices, and weights.
                - Experiences are expected to be stored in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the importance weights for each sampled experience.
        """

        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.size)
        indices = self.tree.sample(batch_size)

        weights = self.tree.levels[-1][indices] ** -self.beta
        weights /= weights.max()

        # Prevents priorities from being zero
        self.beta = min(self.beta + 2e-7, 1)

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (
            *experiences,
            indices.tolist(),
            weights.tolist(),
        )

    def update_priorities(self, indices, priorities):
        """
        Update the priorities of the replay buffer at the given indices.

        Parameters:
        - indices (array-like): The indices of the replay buffer to update.
        - priorities (array-like): The new priorities to assign to the specified indices.

        Returns:
        None
        """
        self.max_priority = max(priorities.max(), self.max_priority)
        self.tree.batch_set(indices, priorities)

    def flush(self):
        """
        Flushes the memory buffers and returns the experiences in order.

        Returns:
            experiences (list): The full memory buffer in order.
        """
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[0 : self.size].tolist())
        self.clear()
        return experiences

    def clear(self):
        """
        Clears the prioritised replay buffer.

        Resets the pointer, size, memory buffers, sum tree, max priority, and beta values.
        """
        self.ptr = 0
        self.size = 0
        self.memory_buffers = []

        self.tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4

    def sample_consecutive(self, batch_size):
        """
        For training MBRL to predict rewards. The right next transition is
        sampled as well. WHEN THE BUFFER IS NOT SHUFFLED.

        It is not overriding the original sample() due to sample() can be used
        for normal training, and the current sample_consecutive() is slower.

        Assumed Add sequence: (State, action, reward, next_state, next_action, next_reard)

        """
        max_length = self.ptr - 1
        candi_indices = list(range(max_length))
        batch_size = min(batch_size, max_length)
        # A list of candidate indices includes all indices.
        sampled_indices = []  # randomly sampled indices that is okay.
        # In this way, the sampling time depends on the batch size rather than buffer size.
        first_sample = True  # Not check duplicate for first time sample.
        while True:
            # Sample size based on how many still needed.
            idxs = random.sample(candi_indices, batch_size - len(sampled_indices))
            for i in idxs:
                # Check if it is already sampled.
                already_sampled = False
                # Only check if it is not first time in the while loop.
                if not first_sample:
                    # compare with each item in the sampled.
                    for j in sampled_indices:
                        if j == i:
                            already_sampled = True
                if (self.memory_buffers[4][i] is False) and (not already_sampled):
                    sampled_indices.append(i)
                if len(sampled_indices) == batch_size:
                    break
            first_sample = False
            if len(sampled_indices) == batch_size:
                break

        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[np.array(sampled_indices)].tolist())

        next_sampled_indices = [x + 1 for x in sampled_indices]
        next_sampled_indices = np.array(next_sampled_indices)
        experiences.append(self.memory_buffers[1][next_sampled_indices].tolist())
        experiences.append(self.memory_buffers[2][next_sampled_indices].tolist())
        return experiences

    def get_statistics(self):
        """
        Compute the statisitics for world model state normalization.
        state, action, reward, next_state, done

        Assumed Add sequence: (State, action, reward, next_state, next_action, next_reard)

        :return: statistic tuple of the collected transitions.
        """

        states = np.array(self.memory_buffers[0][: self.ptr].tolist())
        next_states = np.array(self.memory_buffers[3][: self.ptr].tolist())
        delta = next_states - states

        # Add a small number to avoid zeros.
        observation_mean = np.mean(states, axis=0) + 0.00001
        observation_std = np.std(states, axis=0) + 0.00001
        delta_mean = np.mean(delta, axis=0) + 0.00001
        delta_std = np.std(delta, axis=0) + 0.00001

        statistics = {
            "observation_mean": observation_mean,
            "observation_std": observation_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        return statistics
