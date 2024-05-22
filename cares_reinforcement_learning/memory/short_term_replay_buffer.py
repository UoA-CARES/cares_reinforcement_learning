import random

import numpy as np

from cares_reinforcement_learning.memory import SumTree
from collections import deque


class ShortTermReplayBuffer:
   
    def __init__(self, max_capacity: int = int(1e6), **memory_params):
        self.memory_params = memory_params

        self.max_capacity = max_capacity
        self.memory_buffers = deque([], maxlen=self.max_capacity)
  

    def add(self, state, action, reward, next_state, done, episode_num, episode_step) -> None:
      
        experience = [state, action, reward, next_state, done, episode_num, episode_step]
        
        self.memory_buffers.append(experience)

    
    
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
        
        # Randomly sample an experience
        based_exp = random.sample(self.memory_buffers, 1)
        randomExperience = based_exp[0]
        state, action, reward, next_state, done, episode_num, episode_step = randomExperience
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = self.sample_episode(episode_num, episode_step, batch_size)
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    
    def sample_episode(self, target_episode_num: int, target_episode_step: int, batch_size: int) -> tuple:
       
        matching_index = None
       
        # Find the matching index
        for i, experience in enumerate(self.memory_buffers):
           # print(f"experience:{experience}, i:{i}")
            #input()
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
      
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self.memory_buffers)