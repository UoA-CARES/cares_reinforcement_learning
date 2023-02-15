
import copy

import torch
import numpy as np


class SAC:

    def __init__(self,
                 actor_network,
                 critic_one,
                 critic_two,
                 max_actions,
                 min_actions,
                 gamma,
                 tau,
                 alpha,
                 device):

        self.actor_net = actor_network.to(device)

        self.critic_one_net = critic_one.to(device)
        self.target_critic_one_net = copy.deepcopy(critic_one).to(device)

        self.critic_two_net = critic_two.to(device)
        self.target_critic_two_net = copy.deepcopy(critic_two).to(device)

        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.max_actions = torch.FloatTensor(max_actions).to(device)
        self.min_actions = torch.FloatTensor(min_actions).to(device)

        self.learn_counter = 0
        self.policy_update_freq = 2  # Hard coded

        self.device = device

    def forward(self, observation):
        pi_action, logp_pi = self.actor_net.forward(observation)
        return pi_action, logp_pi

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

        with torch.no_grad():
            """
            Unlike in TD3, the next-state actions used in the target come from the current policy instead of a target policy.
            next_actions = self.target_actor_net(next_states).to(self.device) #### retrieve ā~π(.|s') from actor network
            """
            next_actions, logp_pi_batch = self.actor_net(next_states)

            next_actions = next_actions.to(self.device)
            logp_pi_batch = logp_pi_batch.to(self.device).unsqueeze(1)

            target_q_values_one = self.target_critic_one_net(next_states, next_actions)
            target_q_values_two = self.target_critic_two_net(next_states, next_actions)

            target_q_values = torch.min(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * (target_q_values - self.alpha * logp_pi_batch)

        q_values_one = self.critic_one_net(states, actions)
        q_values_two = self.critic_two_net(states, actions)

        # Update the Critic One
        critic_one_loss = self.critic_one_net.loss(q_values_one, q_target)

        self.critic_one_net.optimiser.zero_grad()
        critic_one_loss.backward()
        self.critic_one_net.optimiser.step()

        # Update Critic Two
        critic_two_loss = self.critic_two_net.loss(q_values_two, q_target)

        self.critic_two_net.optimiser.zero_grad()
        critic_two_loss.backward()
        self.critic_two_net.optimiser.step()

        # updating the target network and the actor network
        if self.learn_counter % self.policy_update_freq == 0:

            # Update Actor
            pi_action, logp_pi_batch = self.actor_net(states)
            logp_pi_batch = logp_pi_batch.to(self.device).unsqueeze(1)

            actor_q_one = self.critic_one_net(states, pi_action)
            actor_q_two = self.critic_two_net(states, pi_action)

            actor_q_values = torch.min(actor_q_one, actor_q_two)

            actor_loss = -(actor_q_values - self.alpha * logp_pi_batch).mean()

            # Update Actor network parameters
            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net.optimiser.step()

            # Update target network params with an exponential filter
            for target_param, param in zip(self.target_critic_one_net.parameters(), self.critic_one_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_critic_two_net.parameters(), self.critic_two_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))



