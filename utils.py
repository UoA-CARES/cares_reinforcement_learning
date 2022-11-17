from collections import deque
import random
import torch
import numpy as np

def train(agent, episode_num, batch_size, env):
    
    # Track the reward over EPISODE_NUM episodes
    historical_reward = []
    
    for episode in range(0, episode_num):
        
        # Initial State
        state, _ = env.reset()

        episode_reward = 0
        
        while True:
            
            action = agent.choose_action(state)

            # Take the next action and observe the effect
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            # Add the experience to the memory buffer
            agent.memory.add(state, action, reward, next_state, terminated)
            
            # Train the neural network when the size of the memory buffer is greater than or equal to the batch size
            for _ in range(1, 10):
                agent.learn(batch_size)

            state = next_state
            episode_reward += reward
            
            if(terminated or truncated):
                break
        
        historical_reward.append(episode_reward)
        
        print(f"Episode #{episode} Reward {episode_reward}")

    # Data collected during run, for plotting
    episode_data = range(0, episode_num)
    reward_data = historical_reward

    return (episode_data, reward_data)

class MemoryBuffer:

    def __init__(self, max_capacity):
        self.buffer = deque([],maxlen=max_capacity)
        
    def add(self, *experience):
        self.buffer.append(experience)
        
    def sample(self, batch_size):
        
        # Randomly sample experiences from buffer of size batch_size
        experienceBatch = random.sample(self.buffer, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        states, actions, rewards, next_states, dones = zip(*experienceBatch)
        
        # Convert from _ tuples to _ tensors
        # eg. states tuple to states tensor
        states = torch.tensor(np.asarray(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.asarray(next_states), dtype=torch.float32)
        dones = torch.tensor(dones)
        
        return (states, actions, rewards, next_states, dones)