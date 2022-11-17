import torch
import random

from typing import Tuple
from .utils import MemoryBuffer
from gym import Space

class DQNAgent(object):
    
    def __init__(self, network : torch.nn.Module, memory : MemoryBuffer, epsilon_info : Tuple[float, float, float], gamma : float, action_space : Space):
        self.memory = memory
        self.network = network

        self.epsilon, self.min_epsilon, self.epsilon_decay = epsilon_info

        self.gamma = gamma
        self.action_space = action_space
    
    def choose_action(self, state):
        
        # With probability Epsilon, select a random action
        if random.random() < self.epsilon:
            return self.action_space.sample()
        
        # Generate Expected Future Return
        state_tensor = torch.tensor(state)
        q_values = self.network.forward(state_tensor)
        
        # Select the action with the best estimated future return
        return torch.argmax(q_values).item()

    def learn(self, batch_size : int):

        # Only begin learning when there is enough experience in buffer to sample from
        if len(self.memory.buffer) < batch_size:
            return
        
        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)
        
        # Generate Q Values given state at time t and t + 1 
        q_values = self.network.forward(states)
        next_q_values = self.network.forward(next_states)
        
        # Get the q values using current model of the actual actions taken historically
        best_q_values = q_values[torch.arange(q_values.size(0)), actions]
        
        # For q values at time t + 1, return all the best actions in each state
        best_next_q_values = torch.max(next_q_values, dim=1).values
        
        # Compute the target q values based on bellman's equations
        expected_q_values = rewards + self.gamma * best_next_q_values * ~dones
        
        # Update the Network
        loss = self.network.loss(best_q_values, expected_q_values)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()
        
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)