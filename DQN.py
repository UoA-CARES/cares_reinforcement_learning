import torch
import random

from typing import Tuple
from .utils import MemoryBuffer
from gym import Space


class DQNAgent(object):
    """
    Reinforcement Learning agent using DQN algorithm to learn
    """

    def __init__(self, network: torch.nn.Module, memory: MemoryBuffer, epsilon_info: Tuple[float, float, float],
                 gamma: float, action_space: Space):
        """
        Parameters
            `network`: neural network used for Q value estimation
            `memory`: buffer used to store experience/transitions
            `epsilon_info`: tuple containing the epsilon max, min, decay.
                Formatted as `(EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY)`
            `gamma`: discount rate
            `action_space`: the action space of the environment
        """
        self.memory = memory
        self.network = network

        self.epsilon, self.min_epsilon, self.epsilon_decay = epsilon_info

        self.gamma = gamma
        self.action_space = action_space

    def choose_action(self, state):
        """
        Use epsilon greedy policy to select the next action

        Parameters
            `state`: observation to be used to decide the next action

        Returns
            `action`: see the action_space to determine type
        """
        # With probability Epsilon, select a random action
        if random.random() < self.epsilon:
            return self.action_space.sample()

        # Generate Expected Future Return
        state_tensor = torch.tensor(state)
        q_values = self.network.forward(state_tensor)

        # Select the action with the best estimated future return
        return torch.argmax(q_values).item()

    def learn(self, batch_size: int) -> None:
        """
        Initiates experience replay

        Parameters:
            `batch_size`: the size of the batch to be sampled
        """

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


class DoubleDQNAgent(object):
    """
    Reinforcement learning agent using Double DQN to learn
    """

    def __init__(self, main_network: torch.nn.Module, target_network: torch.nn.Module, memory: MemoryBuffer,
                 epsilon_info: Tuple[float, float, float], gamma: float, tau: float, action_space: Space) -> None:

        """
        Parameters
            `main_network`: network used for Q value estimation
            `target_network`: network used to calculate Target Q
            `memory`: buffer used to store experience/transitions
            `epsilon_info`: tuple containing the epsilon max, min, decay.
                Formatted as `(EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY)`
            `gamma`: discount rate
            `tau`: polyak averaging constant for parameter copying
            `action_space`: the action space of the environment
        """
        self.memory = memory
        self.q_net = main_network
        self.q_net_prime = target_network

        self.action_space = action_space

        self.epsilon, self.min_epsilon, self.epsilon_decay = epsilon_info
        self.gamma = gamma
        self.tau = tau

    def choose_action(self, state):
        """
        Use epsilon greedy policy to select the next action

        Parameters
            `state`: observation to be used to decide the next action

        Returns
            `action`: see the action_space to determine type
        """

        # With probability Epsilon, select a random action
        if random.random() < self.epsilon:
            return self.action_space.sample()

        # Generate Expected Future Return
        state_tensor = torch.tensor(state)
        q_values = self.q_net.forward(state_tensor)

        # Select the action with the best estimated future return
        return torch.argmax(q_values).item()

    def learn(self, batch_size):
        """
        Initiates experience replay

        Parameters:
            `batch_size`: the size of the batch to be sampled
        """

        # Being the learning process when the number of experiences in the buffer is greater than the batch sized desired
        if len(self.memory.buffer) < batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample(batch_size)

        q_values = self.q_net(states)
        next_q_values_prime = self.q_net_prime(next_states)

        '''
        Selected using the Q values generated by the main network (Q)
        Returns a 1D tensor of the q values reflecting the actions that the agent took at that state in the past
        '''
        q_values_of_actions_taken = q_values[torch.arange(q_values.size(0)), actions]

        '''
        Select the best action to take, given the q values generated by the target network (Q')
        Returns a 1D tensor of the best action to take given the state
        '''
        actions_prime = torch.argmax(next_q_values_prime, dim=1)

        next_q_values = self.q_net(next_states)
        '''
        Using the actions_prime generated by the target network, select the q values of taking action prime (a') 
        generated using the target network, given the state
        Returns a 1D Tensor of q values generated using the main network (Q) at time T + 1 given action prime (a')
        '''
        next_q_values_of_actions_taken = next_q_values[torch.arange(next_q_values.size(0)), actions_prime]

        q_target = rewards + self.gamma * next_q_values_of_actions_taken * ~dones

        loss = self.q_net.loss(q_values_of_actions_taken, q_target)
        self.q_net.optimizer.zero_grad()
        loss.backward()
        self.q_net.optimizer.step()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)

        for target_param, param in zip(self.q_net_prime.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
