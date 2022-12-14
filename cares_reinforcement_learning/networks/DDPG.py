
import torch
import torch.nn as nn
import numpy as np
import copy


class DDPG:

    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 device
                 ):

        self.actor_net: nn.Module = actor_network.to(device)
        self.targ_actor_net: nn.Module = copy.deepcopy(actor_network)

        self.critic_net: nn.Module = critic_network.to(device)
        self.targ_critic_net: nn.Module = copy.deepcopy(critic_network)

        self.gamma = gamma
        self.tau = tau

        self.device = device

    def forward(self, observation):
        return self.actor_net.forward(observation)

    def learn(self, experiences):

        states, actions, rewards, next_states, dones = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(dones).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # We do not want the gradients calculated for any of the target old_networks, we manually update the parameters
        with torch.no_grad():

            next_actions = self.targ_actor_net(next_states).to(self.device)
            next_q_values = self.targ_critic_net(next_states, next_actions)

            q_target = rewards + self.gamma * (1 - dones) * next_q_values

        q_values = self.critic_net(states, actions)

        # Update the Critic Network
        critic_loss = self.critic_net.loss(q_values, q_target)

        self.critic_net.optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net.optimiser.step()

        # Update the Actor Network
        actor_q = self.critic_net(states, self.actor_net(states).to(self.device))
        actor_loss = -actor_q.mean()

        self.actor_net.optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net.optimiser.step()

        # Update target old_networks' params
        for target_param, param in zip(self.targ_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.targ_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
