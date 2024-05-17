import random

import numpy as np

from cares_reinforcement_learning.memory import SumTree
from collections import deque


class LongTermReplayBuffer:
    """
    A prioritized replay buffer implementation for reinforcement learning.

    This buffer stores experiences and allows for efficient sampling based on priorities.
    Experiences are stored in the order: state, action, reward, next_state, done, ...

    Args:
        max_capacity (int): The maximum capacity of the buffer.
        **priority_params: Additional parameters for priority calculation.

    Attributes:
        priority_params (dict): Additional parameters for priority calculation.
        max_capacity (int): The maximum capacity of the buffer.
        current_size (int): The current size of the buffer.
        memory_buffers (list): An array of buffers for each experience type.
        tree (SumTree): The SumTree data structure for efficient sampling based on priorities.
        tree_pointer (int): The location to add the next item into the tree.
        max_priority (float): The maximum priority value in the buffer.
        beta (float): The beta parameter for importance weight calculation.

    Methods:
        __len__(): Returns the current size of the buffer.
        add(state, action, reward, next_state, done, *extra): Adds a single experience to the buffer.
        sample_uniform(batch_size): Samples experiences uniformly from the buffer.
        sample_priority(batch_size): Samples experiences from the buffer based on priorities.
        sample_inverse_priority(batch_size): Samples experiences from the buffer based on inverse priorities.
        update_priorities(indices, priorities): Updates the priorities of the buffer at the given indices.
        flush(): Flushes the memory buffers and returns the experiences in order.
        sample_consecutive(batch_size): Randomly samples consecutive experiences from the memory buffer.
    """

    def __init__(self, max_capacity: int = int(1e3), **priority_params):
        self.priority_params = priority_params

        self.max_capacity = max_capacity

        # size is the current size of the buffer
        self.current_size = 0

        # Functionally is an array of buffers for each experience type
        self.memory_buffers = deque([], maxlen=self.max_capacity)
        self.full = False
        self.deleted_episode_id = 0
        self.min_reward = 1e6
        self.min_id = None
        
        
        # The priority arguments in the buffer

        self.max_priority = 1.0
        self.beta = 0.4

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.current_size

    def add(self,episode_id, episode_reward) -> None:
        """
        Adds a single experience to long term memory buffer.

        Data is expected to be stored in the order: episode_id, episode_reward.
        In this buffer we store experiences that gain higher rewards.
    
        Args:
            episode_id (int): The episode id.
            episode_reward (float): The episode reward.

        Returns:
            None
        """
        experience = [episode_id, episode_reward]
        if self.is_full():
            if episode_reward > self.min_reward:
                # Find the index of the experience with the minimum reward
                #print(f"episode_id: {episode_id}, min_id:{self.min_id}")
                min_index = next(i for i, (ep_id, ep_reward) in enumerate(self.memory_buffers) if ep_id == self.min_id)
                self.deleted_episode_id = min_index

                # Replace the experience with the minimum reward with the current one
                self.buffer[min_index] = experience

                # Update min_reward and min_id with the new minimum values
                self.min_reward = min(self.memory_buffers, key=lambda x: x[1])[1]
                self.min_id = min(self.m, key=lambda x: x[1])[0]
        else:
            # If the buffer is not full, simply append the experience
            self.memory_buffers.append(experience)

            self.current_size = min(self.current_size + 1, self.max_capacity)
            #print(f"episode_reward: {episode_reward}, min_id:{self.min_id}, min_reward:{self.min_reward}")
            if episode_reward < self.min_reward:
                self.min_reward = episode_reward
                self.min_id = episode_id

            # Check if the buffer is full now
            if len(self.memory_buffers) == self.max_capacity:
                self.full = True
            else:
                self.full = False  # Set to False in case it was previously True
                
        
    def is_full(self) -> bool:
        """
        Returns whether the long term memory buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        """
        return self.full

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

        episode_ids, _ = zip(*experiences)
        # print(f"episode_id:{experiences[0]}")

        return episode_ids
    def get_min_reward(self) -> float:
        """
        Returns the minimum reward in the long term memory buffer.

        Returns:
            float: The minimum reward in the buffer.
        """
        return self.min_reward
    
    def get_deleted_episode_id(self) -> int:
        """
        Returns the episode id of the deleted episode in the long term memory buffer.

        Returns:
            int: The episode id of the deleted episode.
        """
        return self.deleted_episode_id
    
    def clear(self) -> None:
        
        self.memory_buffers = deque([], maxlen=self.max_capacity)
        self.full = False
        self.deleted_episode_id = 0
        self.min_reward = 1e6
        self.min_id = None
        self.current_size = 0
        

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

    # def clear(self) -> None:
    #     """
    #     Clears the prioritised replay buffer.

    #     Resets the pointer, size, memory buffers, sum tree, max priority, and beta values.
    #     """
    #     self.tree_pointer = 0
    #     self.current_size = 0
    #     self.memory_buffers = []

    #     self.tree = SumTree(self.max_capacity)
    #     self.max_priority = 1.0
    #     self.beta = 0.4

    #********** It is wrong , the second argument should be the episode_reward
    def calculate_euclidean_distances(self, target_state):
        euclidean_distances = []
        for experience in self.memory_buffers:
            state = experience[1]
            # Calculate Euclidean distance
            distance = np.linalg.norm(target_state - state)  
            
            # Store episode_id and distance
            euclidean_distances.append((experience[0], distance))  

        # Sort the distances by ascending order
        euclidean_distances.sort(key=lambda x: x[1])

        return euclidean_distances

    def calculate_cosine_distances(self, target_state):
        cosine_distances = []
        for e in self.memory_buffers:
            episode_id, state = e[0], e[1]

            state_norm = np.linalg.norm(state)
            target_state_norm = np.linalg.norm(target_state)

            if state_norm == 0 or target_state_norm == 0:
                distance = 0
            else:
                similarity = np.dot(state, target_state) / (state_norm * target_state_norm)
                distance = 1 - similarity  # Inverse cosine similarity to get a distance-like value

            cosine_distances.append((episode_id, distance))  # Store episode_id and distance

        # Sort the distances based on the distance values and then episode_id
        sorted_cosine_distances = sorted(cosine_distances, key=lambda x: (tuple(x[1]), x[0]))
        return sorted_cosine_distances

    def calculate_mahalanobis_distances(self, target_state, covariance_matrix):
        mahalanobis_distances = []
        for e in self.buffer:
            episode_id, state = e[0], e[1]

            delta = state - target_state
            distance = np.sqrt(np.dot(np.dot(delta, np.linalg.inv(covariance_matrix)), delta))

            mahalanobis_distances.append((episode_id, distance))  # Store episode_id and distance

        sorted_mahalanobis_distances = sorted(mahalanobis_distances, key=lambda x: x[1])

        return sorted_mahalanobis_distances

    def find_related_episodes(self, target_state, target_next_state, num_related_episodes=1, similarity_metric='mahalanobis'):

        if similarity_metric == 'euclidean':
            state_euclidean_distances = self.calculate_euclidean_distances(target_state)
            n_state_euclidean_distances = self.calculate_euclidean_distances(target_next_state)
        elif similarity_metric == 'cosine':
            state_euclidean_distances = self.calculate_cosine_distances(target_state)
            n_state_euclidean_distances = self.calculate_cosine_distances(target_next_state)
        elif similarity_metric == 'mahalanobis':
            covariance_matrix = np.identity(len(target_state))  # Example covariance matrix
            state_euclidean_distances = self.calculate_mahalanobis_distances(target_state, covariance_matrix)
            covariance_next_matrix = np.identity(len(target_next_state))  # Example covariance matrix
            n_state_euclidean_distances = self.calculate_mahalanobis_distances(target_next_state,
                                                                               covariance_next_matrix)
        else:
            raise ValueError("Invalid similarity_metric")

        # Combine the distances from both target_state and target_next_state
        '''combined_distances = state_euclidean_distances + n_state_euclidean_distances

        sorted_combined_distances = sorted(combined_distances, key=lambda x: (tuple(x[1]), x[0]))

        # Find the episode_ids of the most similar episodes based on the combined distances
        similar_episodes = [episode_id for episode_id, _ in sorted_combined_distances[:num_related_episodes]]

        return similar_episodes if similar_episodes else None '''

        combined_distances = state_euclidean_distances + n_state_euclidean_distances

        sorted_combined_distances = sorted(combined_distances, key=lambda x: x[1])

        # Find the episode_ids of the most similar episodes based on the combined distances
        similar_episodes = [episode_id for episode_id, _ in sorted_combined_distances[:num_related_episodes]]

        return similar_episodes if similar_episodes else None