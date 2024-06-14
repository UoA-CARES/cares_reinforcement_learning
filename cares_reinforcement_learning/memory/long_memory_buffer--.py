import random
import numpy as np
from cares_reinforcement_learning.memory.prioritised_replay_buffer import PrioritizedReplayBuffer


class LongMemoryBuffer(PrioritizedReplayBuffer):
    
    def __init__(self, max_capacity: int = int(1e3), **priority_params):
        super().__init__(max_capacity=max_capacity, **priority_params)
        self.min_reward = float('inf')
        self.min_index = -1

    # def add(self, experience) -> None:
    #     episode_reward = experience[1]  # total_reward is at index 1
        
    #     if self.is_full():
    #         if episode_reward > self.min_reward:
    #             self.memory_buffers[self.min_index] = experience
    #             self.update_min_reward()
    #     else:
    #         self.memory_buffers.append(experience)
    #         if episode_reward < self.min_reward:
    #             self.min_reward = episode_reward
    #             self.min_index = len(self.memory_buffers) - 1
                                
  
                
    def add_episode(self, experience) -> None:
        
        episode_reward = experience[1]  # episode_reward is at index 1

        # Check if the buffer is full
        if self.is_full():
            if episode_reward > self.min_reward:
                # Replace the experience with the minimum reward
                for index, exp in enumerate(experience):
                    self.memory_buffers[index][self.min_index] = exp
                # Update the minimum reward and index
                self.update_min_reward()
        else:
            
            # Dynamically create the full memory size on first experience
            for index, exp in enumerate(experience):
                if index >= len(self.memory_buffers):
                    # Create a list of numpy arrays to use index extraction in sample O(1)
                    memory = np.array([None] * self.max_capacity)
                    self.memory_buffers.append(memory)

                # Add the experience to the latest position in the buffer
                self.memory_buffers[index][self.tree_pointer] = exp 
            # Update the minimum reward and index if necessary
            if episode_reward < self.min_reward:
                self.min_reward = episode_reward
                self.min_index = self.tree_pointer

        # Update tree pointer and current size
        self.tree_pointer = (self.tree_pointer + 1) % self.max_capacity
        self.current_size = min(self.current_size + 1, self.max_capacity)
         
    
    def update_min_reward(self):
        if self.memory_buffers:
            rewards = [entry[1] for entry in self.memory_buffers]
            self.min_index = np.argmin(rewards)
            self.min_reward = rewards[self.min_index]
        else:
            self.min_reward = float('inf')
            self.min_index = -1
    
    def is_full(self):
        return len(self.memory_buffers) >= self.max_capacity
    
    def get_length(self):
        return len(self.memory_buffers)
    

    
    def get_min_reward(self) -> float:
        """
        Returns the minimum reward in the long term memory buffer.

        Returns:
            float: The minimum reward in the buffer.
        """
        return self.min_reward
    
    # def get_replaced_episode_id_reward(self) -> int:
    #     """
    #     Returns the episode id of the deleted episode in the long term memory buffer.

    #     Returns:
    #         int: The episode id of the deleted episode.
    #     """
    #     return self.replaced_episode_id, self.replaced_episode_reward
    
   