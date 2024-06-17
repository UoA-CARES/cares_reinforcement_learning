import random

import numpy as np

#from cares_reinforcement_learning.memory import SumTree
from collections import deque


class ShortTermReplayBuffer:
   
    def __init__(self, max_capacity: int = int(1e6), **memory_params):
        self.memory_params = memory_params

        self.max_capacity = max_capacity
        self.memory_buffers = deque([], maxlen=self.max_capacity)
  

    def add(self, state, action, reward, next_state, done, episode_num, episode_step) -> None:
      
        experience = [state, action, reward, next_state, done, episode_num, episode_step]
        
        self.memory_buffers.append(experience)

    def sample_uniform(self, batch_size: int) -> tuple:
        # Randomly sample experiences from buffer of size batch_size
        experience_batch = random.sample(self.memory_buffers, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = zip(*experience_batch)

        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    
    def sample_random_episode(self, batch_size: int) -> tuple:
        
        # Randomly sample an experience
        while True:
            based_exp = random.sample(self.memory_buffers, 1)
            random_experience = based_exp[0]
            state, action, reward, next_state, done, episode_num, episode_step = random_experience
            if episode_step > 1:
                break
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = self.sample_episode(episode_num, episode_step, batch_size)
        # if(episode_step ==2):
        #    print(f"states:{states}, actions:{actions}, rewards:{rewards}, next_states:{next_states}, dones:{dones}, episode_nums:{episode_nums}, episode_steps:{episode_steps}")
        #    input()
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    
    def sample_complete_episode(self, target_episode_num: int, target_episode_step: int) -> tuple:
        
        start_idx = None
        end_idx = None
        
        # Find the start and end indices for the target episode
        for i, experience in enumerate(self.memory_buffers):
            episode_num, episode_step = experience[-2], experience[-1]
            
            if episode_num == target_episode_num:
                if episode_step == 1:
                    start_idx = i
                    end_idx = i+1 
                elif episode_step == target_episode_step:
                    end_idx = i + 1
                    break
        
        # print(f"start_idx:{start_idx}, end_idx:{end_idx}")
        # input()
    
        if start_idx is None or end_idx is None:
            raise ValueError("No matching experience found")

        # Extract the batch of experiences
        experience_batch = list(self.memory_buffers)[start_idx:end_idx]
        # print(f"experience_batch:{experience_batch}")
        # input()
        # Unpack the experiences
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = zip(*experience_batch)
        
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
        if matching_index < batch_size or target_episode_step < batch_size:
            
            start_idx = max(0,matching_index -target_episode_step+1)
            #print(f"start_idx:{start_idx}, matching_index:{matching_index}, target_episode_step:{target_episode_step}")
            
            #end_idx = min(self.max_capacity,matching_index + batch_size)
        else:
            start_idx = max(0, matching_index - batch_size)
            # for i in range(1, batch_size):
            #    if  self.memory_buffers[matching_index - i][-2] != target_episode_num:
            #        start_idx = max(0, matching_index - i+1)
            #        break
        end_idx = matching_index + 1
        # Extract the batch of experiences
        experience_batch = list(self.memory_buffers)[start_idx:end_idx]
       
        if (start_idx == end_idx):
             experience_batch = list(self.memory_buffers)[start_idx:end_idx+1]
        
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = zip(*experience_batch)
        # print(f"start_idx:{start_idx}, end_idx:{end_idx}")
        # print(f"experience_batch:{episode_nums, episode_steps}")
        # input()
      
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return len(self.memory_buffers)