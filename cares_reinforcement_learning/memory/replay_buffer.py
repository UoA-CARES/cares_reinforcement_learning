import random

import numpy as np

from cares_reinforcement_learning.memory import SumTree


class ShortTermReplayBuffer:
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

    def __init__(self, max_capacity: int = int(1e6), **priority_params):
        self.priority_params = priority_params

        self.max_capacity = max_capacity

        # size is the current size of the buffer
        self.current_size = 0

        # Functionally is an array of buffers for each experience type
        self.memory_buffers = []
        # 0 state = []
        # 1 action = []
        # 2 reward = []
        # 3 next_state = []
        # 4 done = []
        # 5 ... = [] e.g. log_prob = []
        # n ... = []

        # The SumTree is an efficient data structure for sampling based on priorities
        self.tree = SumTree(self.max_capacity)
        # The location to add the next item into the tree - index for the SumTree
        self.tree_pointer = 0

        self.max_priority = 1.0
        self.beta = 0.4

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.current_size

    def add(self, state, action, reward, next_state, done, episode_num, episode_step, *extra) -> None:
        """
        Adds a single experience to the prioritized replay buffer.

        Data is expected to be stored in the order: state, action, reward, next_state, done, ...

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.
            reward: The reward received for taking the action.
            next_state: The next state of the environment after taking the action.
            done: A flag indicating whether the episode is done after taking the action.
            *extra: Extra is a variable list of extra experience data to be added (e.g. log_prob).

        Returns:
            None
        """
        experience = [state, action, reward, next_state, done, episode_num, episode_step *extra]

        # Iterate over the list of experiences (state, action, reward, next_state, done, ...) and add them to the buffer
        for index, exp in enumerate(experience):
            # Dynamically create the full memory size on first experience
            if index >= len(self.memory_buffers):
                # NOTE: This is a list of numpy arrays in order to use index extraction in sample O(1)
                memory = np.array([None] * self.max_capacity)
                self.memory_buffers.append(memory)

            # This adds to the latest position in the buffer
            self.memory_buffers[index][self.tree_pointer] = exp

        self.tree.set(self.tree_pointer, self.max_priority)

        self.tree_pointer = (self.tree_pointer + 1) % self.max_capacity
        self.current_size = min(self.current_size + 1, self.max_capacity)

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
    
    def sample_random_episode(self, batch_size: int) -> tuple:
        """
        Randomly samples an episode from the memory buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences and their corresponding indices.
                - Experiences are returned in the order: state, action, reward, next_state, done, ...
                - The indices represent the indices of the sampled experiences in the buffer.
        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)
        random_indice = np.random.randint(self.current_size, size=1)
        randomExperience = self.memory_buffers[0][random_indice]
        state, action, reward, next_state, done, episode_num, episode_step = randomExperience
        episode_id, reward_matching_index, states, actions, rewards, next_states, dones, episode_nums, episode_steps = self.sample_episode(episode_num, episode_step, batch_size)
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    
    def sample_episode(self, target_episode_num: int, target_episode_step: int, batch_size: int) -> tuple:
        """

        Args:
            target_episode_num (int): _description_
            target_episode_step (int): _description_
            batch_size (int): _description_

        Returns:
            tuple:
        """
        matching_index = None

        # Find the matching index
        for i, experience in enumerate(self.memory_buffers):
            episode_num, episode_step = experience[-2], experience[-1]
            if episode_num == target_episode_num and episode_step == target_episode_step:
                matching_index = i
                break
        if matching_index is None:
            raise ValueError("No matching experience found")

        if matching_index >= self.max_capacity or matching_index < 0:
            raise ValueError("Index out of bounds")
        if matching_index < batch_size:
            start_idx = matching_index
            end_idx = min(self.max_capacity,matching_index + batch_size)
        else:
            # Determine the starting and ending indices for the batch
            start_idx = max(0, matching_index - batch_size)
            end_idx = matching_index
        # Extract the batch of experiences
        experience_batch = list(self.memory_buffers)[start_idx:end_idx]
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = zip(*experience_batch)
        episode_id = f"{target_episode_num}_{target_episode_step}"
        # Access the reward of the experience with matching_index
        reward_matching_index = self.memory_buffers[end_idx][2]  # Assuming reward is at index 3

        #print(f"Matching index : {matching_index} reward_matching_index:{reward_matching_index}") #Sampled episode num: {episode_nums}, episode steps: {episode_steps}")

        return episode_id, reward_matching_index, states, actions, rewards, next_states, dones, episode_nums, episode_steps
    
    

        
        
        
        
        
        
        
        
        

        
        

        # Extracts the experiences at the desired indices from the buffer
        experiences = []
        for buffer in self.memory_buffers:
            # NOTE: we convert back to a standard list here
            experiences.append(buffer[indices].tolist())

        return (*experiences, indices.tolist())

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