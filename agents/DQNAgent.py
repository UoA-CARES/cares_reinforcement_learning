import torch
import numpy as np
import random

from gym import Space

from ..util import MemoryBuffer


class DQNAgent(object):
    """
    Reinforcement Learning agent using DQN algorithm to learn
    """

    def __init__(self,
                 network: torch.nn.Module,
                 memory: MemoryBuffer,
                 epsilon_max: float,
                 epsilon_min: float,
                 epsilon_decay: float,
                 gamma: float,
                 action_space: Space):
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

        self.epsilon = epsilon_max
        self.min_epsilon = epsilon_min
        self.epsilon_decay = epsilon_decay

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

        # Convert into Tensor
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.LongTensor(np.asarray(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.LongTensor(dones)

        # Generate Q Values given state at time t and t + 1 
        q_values = self.network.forward(states)
        next_q_values = self.network.forward(next_states)

        # Get the q values using current model of the actual actions taken historically
        best_q_values = q_values[torch.arange(q_values.size(0)), actions]

        # For q values at time t + 1, return all the best actions in each state
        best_next_q_values = torch.max(next_q_values, dim=1).values

        # Compute the target q values based on bellman's equations
        expected_q_values = rewards + self.gamma * (1 - dones) * best_next_q_values

        # Update the Network
        loss = self.network.loss(best_q_values, expected_q_values)
        self.network.optimizer.zero_grad()
        loss.backward()
        self.network.optimizer.step()

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.min_epsilon)

