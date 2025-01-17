import random

import numpy as np
from collections import deque
import heapq
from sklearn.neighbors import NearestNeighbors
class ValuableEpisodeBuffer:
  

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
    
    def fetch_nearest_episode_euclidean_distance(self, current_state):
        closest_distance = float('inf')
        closest_episode = None

        for experience in self.memory_buffers:
            # Assuming experience[2] is the goal (x, y)
            distance = np.linalg.norm(np.array(experience[2]) - np.array(current_state))
            if distance < closest_distance:
                closest_distance = distance
                closest_episode = experience

        # Unpack the closest episode
        (
            episode_num, total_reward, episode_first_state, 
            states, actions, rewards, next_states, 
            dones, episode_nums, episode_steps
        ) = closest_episode

        return actions, states, episode_num, episode_steps, rewards, total_reward

    def fetch_k_nearest_episodes_euclidean_distance(self, current_state, k=1):
        """
        Fetch k-nearest episodes based on Euclidean distance.
        
        Args:
            current_state (list or np.array): Current state to compare.
            k (int): Number of nearest neighbors to retrieve.

        Returns:
            list: A list of k nearest episodes.
        """
        # Min-heap to store distances and corresponding experiences
        distance_heap = []
        
        for experience in self.memory_buffers:
            distance = np.linalg.norm(np.array(experience[2]) - np.array(current_state))
            heapq.heappush(distance_heap, (distance, experience))
        
        # Extract k smallest distances (nearest neighbors)
        k_nearest = [heapq.heappop(distance_heap)[1] for _ in range(min(k, len(distance_heap)))]
        return k_nearest
    
    
    
    def fetch_approx_k_nearest(self, current_state, k=1):
        # Ensure there are enough states in the buffer
        if len(self.memory_buffers) == 0:
            raise ValueError("Memory buffer is empty")
        
        states = [experience[2] for experience in self.memory_buffers]
        nbrs = NearestNeighbors(n_neighbors=min(k, len(states)), algorithm='auto').fit(states)
        distances, indices = nbrs.kneighbors([current_state])
        return [self.memory_buffers[idx] for idx in indices[0]]
    
     def fetch_approx_k_nearest_goal(self, current_goal, k=1):
        # Ensure there are enough states in the buffer
        if len(self.memory_buffers) == 0:
            raise ValueError("Memory buffer is empty")
        
        goals = [experience[2] for experience in self.memory_buffers]
        nbrs = NearestNeighbors(n_neighbors=min(k, len(goals)), algorithm='auto').fit(goals)
        distances, indices = nbrs.kneighbors([current_goal])
        return [self.memory_buffers[idx] for idx in indices[0]]
    
    def fetch_nearest_episode(self, current_state, k=3):
        # Get the nearest episode using k-NN
        nearest_experiences = self.fetch_approx_k_nearest(current_state, k=k) # k = 1, 3, 5, 10
        if not nearest_experiences:
            raise ValueError("No nearest episode found")
        
        # Extract the closest episode
        closest_episode = nearest_experiences[0]
        
        (
            episode_num, total_reward, episode_first_state, states, actions,
            rewards, next_states, dones, episode_nums, episode_steps
        ) = closest_episode

        # Return the required fields
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
     def fetch_nearest_goal_episode(self, current_goal, k=5):
        # Get the nearest episode using k-NN
        nearest_experiences = self.fetch_approx_k_nearest_goal(current_goal, k=k) # k = 1, 3, 5, 10
        if not nearest_experiences:
            raise ValueError("No nearest episode found")
        
        # Extract the closest episode
        closest_episode = nearest_experiences[0]
        
        (
            episode_num, total_reward, episode_goal, states, actions,
            rewards, next_states, dones, episode_nums, episode_steps
        ) = closest_episode

        # Return the required fields
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
    def fetch_random_nearest_episode(self, current_state, k=3): # k = 1, 3, 5, 10
        # Get the nearest neighbors using k-NN
        nearest_experiences = self.fetch_approx_k_nearest(current_state, k=k)
        if not nearest_experiences:
            raise ValueError("No nearest episodes found")
        
        # Randomly select one episode from the nearest neighbors
        selected_episode = random.choice(nearest_experiences)
        
        # Extract the fields from the selected episode
        (
            episode_num, total_reward,  episode_first_state, states, actions,
            rewards, next_states, dones, episode_nums, episode_steps
        ) = selected_episode

        # Return the required fields
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
    def fetch_highest_reward_nearest_episode(self, current_state, k=3): # k = 1, 3, 5, 10
        # Get the nearest neighbors using k-NN
        nearest_experiences = self.fetch_approx_k_nearest(current_state, k=k)
        if not nearest_experiences:
            raise ValueError("No nearest episodes found")
        
        # Sort the nearest experiences based on the total reward (descending order)
        nearest_experiences.sort(key=lambda x: x[1], reverse=True)  # x[1] is total_reward
        
        # Select the episode with the highest total reward
        selected_episode = nearest_experiences[0]
        
        # Extract the fields from the selected episode
        (
            episode_num, total_reward, episode_first_state, states, actions,
            rewards, next_states, dones, episode_nums, episode_steps
        ) = selected_episode

        # Return the required fields
        return actions, states, episode_num, episode_steps, rewards, total_reward
    
   

    def fetch_high_rewards_nearest_episode(self, current_state, k=3, selection_strategy="random", near_max_threshold=0.05):
        """
        Fetch one of the episodes with the highest or near-highest reward based on a specified strategy.

        Args:
            current_state: The current state of the environment.
            k (int): The number of nearest neighbors to consider.
            selection_strategy (str): Strategy to select among episodes ('random', 'shortest', 'longest', 'first').
            near_max_threshold (float): Fractional margin to include near-highest reward episodes (default 5%).

        Returns:
            A tuple containing actions, states, episode_num, episode_steps, rewards, and total_reward.
        """
        # Get the nearest neighbors using k-NN
        nearest_experiences = self.fetch_approx_k_nearest(current_state, k=k)
        if not nearest_experiences:
            raise ValueError("No nearest episodes found")

        # Find the maximum total reward
        max_reward = max(nearest_experiences, key=lambda x: x[1])[1]  # x[1] is total_reward

        # Define the near-maximum threshold
        reward_threshold = max_reward * (1 - near_max_threshold)

        # Filter episodes within the near-maximum reward range
        near_max_episodes = [episode for episode in nearest_experiences if episode[1] >= reward_threshold]

        # Select one episode based on the selection strategy
        if selection_strategy == "random":
            selected_episode = random.choice(near_max_episodes)
        elif selection_strategy == "shortest":
            selected_episode = min(near_max_episodes, key=lambda x: x[-1])  # x[-1] is episode_steps
        elif selection_strategy == "longest":
            selected_episode = max(near_max_episodes, key=lambda x: x[-1])  # x[-1] is episode_steps
        elif selection_strategy == "first":
            selected_episode = near_max_episodes[0]
        else:
            raise ValueError(f"Unknown selection strategy: {selection_strategy}")

        # Extract only the required fields from the selected episode
        (
            episode_num, total_reward, episode_first_state, states, actions,
            rewards, next_states, dones, episode_nums, episode_steps
        ) = selected_episode

        # Return the required fields
        return actions, states, episode_num, episode_steps, rewards, total_reward

    



    
    
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
    
    def fetch_nearest_episode(self,current_goal):
        # print(f"episode_num:{episode_num}, episode_steps:{episode_steps}")
        closest_distance = float('inf')
        for experience in self.memory_buffers:
            distance = np.linalg.norm(np.array(experience[2]) - np.array(current_goal))
            if distance < closest_distance:
                closest_distance = distance
                closest_experience = experience
                closest_goal = experience[2]
                return closest_experience, closest_goal


    