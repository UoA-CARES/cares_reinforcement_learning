import random

import numpy as np

from cares_reinforcement_learning.memory import SumTree


class EpisodicBuffer:
 

    def __init__(self, max_capacity: int = int(1e6)):
        self.max_capacity = max_capacity

        # size is the current size of the buffer
        self.current_size = 0

        # Functionally is an array of buffers for each experience type
        self.memory_buffers = []
        self.tree_pointer = 0

        
        #episodic parameters
        self.full = False
   

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.current_size

    def add(self, episode_id, state, action, reward, next_state, done, episode_num, episode_step, *extra) -> None:
      
        experience = [episode_id, state, action, reward, next_state, done, episode_num, episode_step, *extra]

        # Iterate over the list of experiences (state, action, reward, next_state, done, ...) and add them to the buffer
        for index, exp in enumerate(experience):
            # Dynamically create the full memory size on first experience
            if index >= len(self.memory_buffers):
                # NOTE: This is a list of numpy arrays in order to use index extraction in sample O(1)
                memory = np.array([None] * self.max_capacity)
                self.memory_buffers.append(memory)

            # This adds to the latest position in the buffer
            self.memory_buffers[index][self.tree_pointer] = exp

        self.tree_pointer = (self.tree_pointer + 1) % self.max_capacity
        self.current_size = min(self.current_size + 1, self.max_capacity)
        
    def replaceEpisode(self, episode_id, episodic_batch):
        """
        Replace an episode in the episodic memory with a new episode.
        """
        self.delete_episode(episode_id)
        # Replace the episode with the new episode
        for experience in episodic_batch:
            self.add(*experience)
    
    def delete_episode(self, episode_id):
        """
        Delete an episode from the episodic memory.
        """
        # Filter and keep episodes that do not match the target episode_id
        self.memory_buffers = [episode for episode in self.memory_buffers if episode[0] != episode_id]
        
    def sample_episode(self, target_episode_id, batch_size):
        """
        Sample a batch of experiences with a specific episode ID from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.
            target_episode_id (int): Episode ID to sample from.

        Returns:
            list: List of sampled experiences.
        """
        matching_experiences = [experience for experience in self.memory_buffers if experience[0] == target_episode_id]
        
        if len(matching_experiences) >= batch_size:
            sampled_experiences = random.sample(matching_experiences, batch_size)
            # Sort experiences by episode_step in descending order
            sorted_experiences = sorted(matching_experiences, key=lambda x: x['episode_step'], reverse=True)

            # Select the top batch_size experiences with the highest episode_step
            sampled_experiences = sorted_experiences[:batch_size]
        else:
            sampled_experiences = matching_experiences
        
        
        # Unzip the experiences into separate lists
        _, states, actions, rewards, next_states, dones, episode_nums, episode_steps = zip(*sampled_experiences)
        return  states, actions, rewards, next_states, dones, episode_nums, episode_steps

    def is_full(self):
        """
        Check if the episodic memory is full.
        """
        return self.current_size >= self.max_capacity
      

    def sample_uniform(self, batch_size: int) -> tuple:
        """
        Samples experiences uniformly from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences and their corresponding indices.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)
        indices = np.random.randint(self.current_size, size=batch_size)

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (*experiences, indices.tolist())

    def sample_priority(self, batch_size: int) -> tuple:
        """
        Samples experiences from the prioritized replay buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences, indices, and weights.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the importance weights for each sampled experience.
        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)
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

    def sample_inverse_priority(self, batch_size: int) -> tuple:
        """
        Samples experiences from the buffer based on inverse priorities.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences, indices, and weights.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the inverse importance weights for each sampled experience.

        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        top_value = self.tree.levels[0][0]

        # Inverse based on paper for LA3PD - https://arxiv.org/abs/2209.00532
        reversed_priorities = top_value / (
            self.tree.levels[-1][: self.current_size] + 1e-6
        )

        inverse_tree = SumTree(self.max_capacity)

        inverse_tree.batch_set(np.arange(self.tree_pointer), reversed_priorities)

        indices = inverse_tree.sample(batch_size)

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (
            *experiences,
            indices.tolist(),
            reversed_priorities[indices].tolist(),
        )

    def update_priorities(self, indices: list[int], priorities: list[float]) -> None:
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

    def flush(self) -> list[tuple]:
        """
        Flushes the memory buffers and returns the experiences in order.

        Returns:
            experiences (list): The full memory buffer in order.
        """
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[0 : self.current_size].tolist())
        self.clear()
        return experiences

    def sample_consecutive(self, batch_size: int) -> tuple:
        """
        Randomly samples consecutive experiences from the memory buffer.

        Args:
            batch_size (int): The number of consecutive experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences_t and experiences_t+1 and their corresponding indices.
                - Experiences are returned in the order: state_i, action_i, reward_i, next_state_i, done_i, ..._i, state_i+1, action_i+1, reward_i+1, next_state_i+1, done_i+1, ..._+i
                - The indices represent the indices of the sampled experiences in the buffer.

        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        candididate_indices = list(range(self.current_size - 1))

        # A list of candidate indices includes all indices.
        sampled_indices = []  # randomly sampled indices that is okay.
        # In this way, the sampling time depends on the batch size rather than buffer size.

        # Add in only experiences that are not done and not already sampled.
        while len(sampled_indices) < batch_size:
            # Sample size based on how many still needed.
            idxs = random.sample(candididate_indices, batch_size - len(sampled_indices))
            for i in idxs:
                # Check the experience is not done and not already sampled.
                done = self.memory_buffers[4][i]
                if (not done) and (i not in sampled_indices):
                    sampled_indices.append(i)

        sampled_indices = np.array(sampled_indices)

        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[sampled_indices].tolist())

        next_sampled_indices = sampled_indices + 1

        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[next_sampled_indices].tolist())

        return (*experiences, sampled_indices.tolist())

    def get_statistics(self) -> dict[str, np.ndarray]:
        """
        Calculate statistics of the replay buffer.

        Returns:
            statistics (dict): A dictionary containing the following statistics:
                - observation_mean: Mean of the observations in the replay buffer.
                - observation_std: Standard deviation of the observations in the replay buffer.
                - delta_mean: Mean of the differences between consecutive observations.
                - delta_std: Standard deviation of the differences between consecutive observations.
        """
        states = np.array(self.memory_buffers[0][: self.current_size].tolist())
        next_states = np.array(self.memory_buffers[3][: self.current_size].tolist())
        diff_states = next_states - states

        # Add a small number to avoid zeros.
        observation_mean = np.mean(states, axis=0) + 0.00001
        observation_std = np.std(states, axis=0) + 0.00001
        delta_mean = np.mean(diff_states, axis=0) + 0.00001
        delta_std = np.std(diff_states, axis=0) + 0.00001

        statistics = {
            "observation_mean": observation_mean,
            "observation_std": observation_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        return statistics

    def clear(self) -> None:
        """
        Clears the prioritised replay buffer.

        Resets the pointer, size, memory buffers, sum tree, max priority, and beta values.
        """
        self.tree_pointer = 0
        self.current_size = 0
        self.memory_buffers = []

        self.tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        self.beta = 0.4
