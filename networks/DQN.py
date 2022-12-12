import torch
import torch.nn as nn
import numpy as np


class DQN:

    # TODO: determine whether to add typing
    def __init__(self,
                 network: nn.Module,
                 gamma):

        self.network = network

        self.gamma = gamma

    def forward(self, observation):
        return self.network.forward(observation)

    # TODO: take destructured arrays as input, or array of arrays
    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

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
        self.network.optimiser.zero_grad()
        loss.backward()
        self.network.optimiser.step()
