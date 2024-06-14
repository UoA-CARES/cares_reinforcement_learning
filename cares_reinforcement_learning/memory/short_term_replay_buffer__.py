import random

import numpy as np

from cares_reinforcement_learning.memory import SumTree
from collections import deque
from cares_reinforcement_learning.memory.prioritised_replay_buffer import PrioritizedReplayBuffer


class ShortTermReplayBuffer(PrioritizedReplayBuffer):
   
    def __init__(self, max_capacity: int = int(1e6), **memory_params):
        super().__init__()
  

    def add(self, state, action, reward, next_state, done, *extra) -> None:
        episode_num = extra[0]  # Assuming episode_num is the first element of extra
        episode_step = extra[1]  # Assuming episode_step is the second element of extra
        
        experience = [state, action, reward, next_state, done, episode_num, episode_step]
        
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

    
    
    def sample_random_episode(self, batch_size: int) -> tuple:
        
        # Randomly sample an experience
        while True:
            indices = np.random.randint(self.current_size, size=1)
            random_experience = self.memory_buffers[indices[0]]
            state, action, reward, next_state, done, episode_num, episode_step = random_experience
            print(f"state:{state}, action:{action}, reward:{reward}, next_state:{next_state}, done:{done}, episode_num:{episode_num}, episode_step:{episode_step}")
            input()
            if episode_step > 1:
                break
        states, actions, rewards, next_states, dones, episode_nums, episode_steps = self.sample_episode(episode_num, episode_step, batch_size)
        # if(episode_step ==2):
        #    print(f"states:{states}, actions:{actions}, rewards:{rewards}, next_states:{next_states}, dones:{dones}, episode_nums:{episode_nums}, episode_steps:{episode_steps}")
        #    input()
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps
    
   
    def sample_episode(self, target_episode_num: int, target_episode_step: int, batch_size: int) -> tuple:
        
       
        matching_index = None
       
        for index in range(self.current_size):
            if (self.memory_buffers[-2][index] == target_episode_num and
                self.memory_buffers[-1][index] == target_episode_step):
                matching_index = index
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
    
