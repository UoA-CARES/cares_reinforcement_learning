import random

import numpy as np
from collections import deque


class LongMemoryBuffer:
  

    def __init__(self, max_capacity: int = int(1e2), **priority_params):
        self.priority_params = priority_params

        self.max_capacity = max_capacity
        self.memory_buffers = deque([], maxlen=self.max_capacity)
        self.min_high_reward = float('inf')
        self.max_reward = -float('inf')
        self.min_index = -1
        # print(f"max_capacity:{max_capacity}")
        # input()

    def add(self, experience) -> None:
       
        episode_reward = experience[1]  # total_reward is at index 1
        if episode_reward > self.max_reward:
            self.max_reward = episode_reward
        
        if self.is_full():
            if episode_reward > self.min_high_reward:
                self.memory_buffers[self.min_index] = experience
                # Update the minimum reward and index
                self.update_min_reward()
        else:
            # Add the new experience if the buffer is not full
            self.memory_buffers.append(experience)
            # Update the minimum reward and index if necessary
            if episode_reward < self.min_high_reward:
                self.min_high_reward = episode_reward
                self.min_index = len(self.memory_buffers) - 1
        # print(f"memory_buffers_len:{len(self.memory_buffers)}, episode_reward:{experience[1]},max_reward:{ self.max_reward}, min_high_reward:{self.min_high_reward}")
        # input()
    
    def update_min_reward(self):
        if self.memory_buffers:
            rewards = [entry[1] for entry in self.memory_buffers]
            self.min_index = np.argmin(rewards)
            self.min_high_reward = rewards[self.min_index]
        else:
            self.min_high_reward = float('inf')
            self.min_index = -1
    
    def is_full(self):
        return len(self.memory_buffers) >= self.max_capacity
    
    def get_length(self):
        return len(self.memory_buffers)
    
    

    def sample_uniform(self, batch_size: int) -> tuple:
        #print(f"memory_buffers_len:{len(self.memory_buffers)}")
        # for experience in self.memory_buffers:
        #     print(f"episode_id:{experience[0]}, episode_reward:{experience[1]}")
        #     input()
       
       # Randomly sample episodes from buffer of size batch_size
        #experience_batch = random.sample(self.memory_buffers, batch_size)
        
        #input()
        selected_experiences = []
        buffer_length = len(self.memory_buffers)

        for _ in range(batch_size):
            index = random.randint(0, buffer_length - 1)
            selected_experiences.append(self.memory_buffers[index])

        return selected_experiences
    
    def sample_max_reward(self) -> tuple:
        
        for experience in self.memory_buffers:
            if experience[1] == self.max_reward:
                # print (f"max_reward:{experience[1]}")
                # input()
                return experience
        return None
    
    def sample_single(self) -> tuple:
        """
        Sample a single random experience from the buffer.
        Returns:
            tuple: A single random experience from the buffer.
        """
        if not self.memory_buffers:
            raise ValueError("Buffer is empty")

        return random.choice(self.memory_buffers)
        
    
    def get_min_reward(self) -> float:
        """
        Returns the minimum reward in the long term memory buffer.

        Returns:
            float: The minimum reward in the buffer.
        """
        return self.min_high_reward
    def is_empty(self):
        return len(self.memory_buffers) == 0
    
    def get_max_crucial_path(self,number_of_crusial_episodes:int):
        # print (f"buffer rewards:{[experience[1] for experience in self.memory_buffers]}")
        # input()
        episode_batch = self.sample_max_reward()
        if(episode_batch is None):
            raise ValueError("No episode with max reward found")
            
        episode_num,total_reward, states, actions,rewards, next_states, dones, episode_nums, episode_steps = episode_batch
        print(f"episode_num:{episode_num} total_reward:{total_reward}, max_reward:{self.max_reward}")
        #input()
        # print(f", rewards:{rewards}, episode_nums:{episode_nums}, episode_steps:{episode_steps}")
        # input()
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
    
    
    
    def get_crucial_path(self,number_of_crusial_episodes:int):
      
        episode_batch = self.sample_single()
       
        if(episode_batch is None):
            raise ValueError("No episode found")
            
        episode_num,total_reward, states, actions,rewards, next_states, dones, episode_nums, episode_steps = episode_batch
        
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
    def sample_neighbour(self, episode_num:int, episode_steps:int):
        # print(f"episode_num:{episode_num}, episode_steps:{episode_steps}")
        # input()
        for experience in self.memory_buffers:
            if experience[0] == episode_num and experience[7] == episode_steps:
                # print (f"max_reward:{experience[1]}")
                # input()
                return experience
        return None
    
    # def get_replaced_episode_id_reward(self) -> int:
    #     """
    #     Returns the episode id of the deleted episode in the long term memory buffer.

    #     Returns:
    #         int: The episode id of the deleted episode.
    #     """
    #     return self.replaced_episode_id, self.replaced_episode_reward
    
class LongMemoryBufferTotal:
  

    def __init__(self, max_capacity: int = int(1e2), **priority_params):
        self.priority_params = priority_params

        self.max_capacity = max_capacity
        self.memory_buffers = deque([], maxlen=self.max_capacity)
        self.min_high_reward = float('inf')
        self.max_reward = -float('inf')
        self.min_index = -1

    def add(self, experience) -> None:
       
        episode_reward = experience[1]  # total_reward is at index 1
        if episode_reward > self.max_reward:
            self.max_reward = episode_reward
        
        if self.is_full():
            if episode_reward > self.min_high_reward:
                self.memory_buffers[self.min_index] = experience
                # Update the minimum reward and index
                self.update_min_reward()
        else:
            # Add the new experience if the buffer is not full
            self.memory_buffers.append(experience)
            # Update the minimum reward and index if necessary
            if episode_reward < self.min_high_reward:
                self.min_high_reward = episode_reward
                self.min_index = len(self.memory_buffers) - 1
        # print(f"memory_buffers_len:{len(self.memory_buffers)}, episode_reward:{experience[1]},max_reward:{ self.max_reward}, min_high_reward:{self.min_high_reward}")
        # input()
    
    def update_min_reward(self):
        if self.memory_buffers:
            rewards = [entry[1] for entry in self.memory_buffers]
            self.min_index = np.argmin(rewards)
            self.min_high_reward = rewards[self.min_index]
        else:
            self.min_high_reward = float('inf')
            self.min_index = -1
    
    def is_full(self):
        return len(self.memory_buffers) >= self.max_capacity
    
    def get_length(self):
        return len(self.memory_buffers)
    
    

    def sample_uniform(self, batch_size: int) -> tuple:
        #print(f"memory_buffers_len:{len(self.memory_buffers)}")
        # for experience in self.memory_buffers:
        #     print(f"episode_id:{experience[0]}, episode_reward:{experience[1]}")
        #     input()
       
       # Randomly sample episodes from buffer of size batch_size
        #experience_batch = random.sample(self.memory_buffers, batch_size)
        
        #input()
        selected_experiences = []
        buffer_length = len(self.memory_buffers)

        for _ in range(batch_size):
            index = random.randint(0, buffer_length - 1)
            selected_experiences.append(self.memory_buffers[index])

        return selected_experiences
    
    def sample_max_reward(self) -> tuple:
        
        for experience in self.memory_buffers:
            if experience[1] == self.max_reward:
                # print (f"max_reward:{experience[1]}")
                # input()
                return experience
        return None
    
    def sample_single(self) -> tuple:
        """
        Sample a single random experience from the buffer.
        Returns:
            tuple: A single random experience from the buffer.
        """
        if not self.memory_buffers:
            raise ValueError("Buffer is empty")

        return random.choice(self.memory_buffers)
        
    
    def get_min_reward(self) -> float:
        """
        Returns the minimum reward in the long term memory buffer.

        Returns:
            float: The minimum reward in the buffer.
        """
        return self.min_high_reward
    
    def get_max_crucial_path(self,number_of_crusial_episodes:int):
        # print (f"buffer rewards:{[experience[1] for experience in self.memory_buffers]}")
        # input()
        episode_batch = self.sample_max_reward()
        if(episode_batch is None):
            raise ValueError("No episode with max reward found")
            
        episode_num,total_reward, states, actions,rewards, next_states, dones, episode_nums, episode_steps = episode_batch
        print(f"episode_num:{episode_num} total_reward:{total_reward}, max_reward:{self.max_reward}")
        #input()
        # print(f", rewards:{rewards}, episode_nums:{episode_nums}, episode_steps:{episode_steps}")
        # input()
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
    def get_crucial_path(self,number_of_crusial_episodes:int):
      
        episode_batch = self.sample_single()
       
        if(episode_batch is None):
            raise ValueError("No episode found")
            
        episode_num,total_reward, states, actions,rewards, next_states, dones, episode_nums, episode_steps = episode_batch
        
        return actions, states, episode_num, episode_steps, rewards, total_reward