import torch
import numpy as np


class DoubleDQN:

    def __init__(self,
                 main_network,
                 target_network,
                 optimiser,
                 loss,
                 gamma,
                 tau):

        self.main_network = main_network
        self.target_network = target_network

        self.optimiser = optimiser
        self.loss = loss

        self.gamma = gamma
        self.tau = tau

    def forward(self, observation):
        return self.main_network.forward(observation)

    def learn(self, experiences):
        states, actions, rewards, next_states, dones = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.LongTensor(np.asarray(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.LongTensor(dones)

        q_values = self.main_network(states)
        next_q_values_prime = self.target_network(next_states)

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

        next_q_values = self.main_network(next_states)
        '''
        Using the actions_prime generated by the target network, select the q values of taking action prime (a') 
        generated using the target network, given the state
        Returns a 1D Tensor of q values generated using the main network (Q) at time T + 1 given action prime (a')
        '''
        next_q_values_of_actions_taken = next_q_values[torch.arange(next_q_values.size(0)), actions_prime]

        q_target = rewards + self.gamma * (1 - dones) * next_q_values_of_actions_taken

        loss = self.loss(q_values_of_actions_taken, q_target)
        self.main_network.optimizer.zero_grad()
        loss.backward()
        self.main_network.optimizer.step()

        for target_param, param in zip(self.target_network.parameters(), self.main_network.parameters()):
            target_param.data.copy_(self.tau * param + (1 - self.tau) * target_param)
