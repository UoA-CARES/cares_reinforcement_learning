import copy
import os
import torch
from torch.distributions.uniform import Uniform
import numpy as np


class TD3:

    def __init__(self,
                 actor_network,
                 critic_one,
                 critic_two,
                 max_actions,
                 min_actions,
                 gamma,
                 tau,
                 device):
        # TODO: check whether each critic needs its parameters
        self.actor_net = actor_network.to(device)
        self.target_actor_net = copy.deepcopy(actor_network).to(device)

        self.critic_one_net = critic_one.to(device)
        self.target_critic_one_net = copy.deepcopy(critic_one).to(device)

        self.critic_two_net = critic_two.to(device)
        self.target_critic_two_net = copy.deepcopy(critic_two).to(device)

        self.gamma = gamma
        self.tau = tau

        self.max_actions = torch.FloatTensor(max_actions).to(device)
        self.min_actions = torch.FloatTensor(min_actions).to(device)

        self.learn_counter = 0
        self.policy_update_freq = 2  # Hard coded

        self.device = device

    def forward(self, state):

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor.to(self.device)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy()

        return action[0]

    def learn(self, experiences):
        self.learn_counter +=1

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

            next_actions = self.target_actor_net(next_states).to(self.device)

            noise = Uniform(-0.5, 0.5).sample(next_actions.size()).to(self.device)

            next_actions = torch.clip(next_actions + noise, self.min_actions,
                                      self.max_actions)

            target_q_values_one = self.target_critic_one_net(next_states, next_actions)
            target_q_values_two = self.target_critic_two_net(next_states, next_actions)

            target_q_values = torch.min(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one = self.critic_one_net(states, actions)
        q_values_two = self.critic_two_net(states, actions)

        # Update the Critic One
        critic_one_loss = self.critic_one_net.loss(q_values_one, q_target)

        self.critic_one_net.optimiser.zero_grad()
        critic_one_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic_one_net.parameters(), clip_value=1.0)
        self.critic_one_net.optimiser.step()

        # Update Critic Two
        critic_two_loss = self.critic_two_net.loss(q_values_two, q_target)

        self.critic_two_net.optimiser.zero_grad()
        critic_two_loss.backward()
        torch.nn.utils.clip_grad_value_(self.critic_two_net.parameters(), clip_value=1.0)
        self.critic_two_net.optimiser.step()

        if self.learn_counter % self.policy_update_freq == 0:

            # Update Actor
            actor_q_one = self.critic_one_net(states, self.actor_net(states))
            actor_q_two = self.critic_two_net(states, self.actor_net(states))

            actor_q_values = torch.min(actor_q_one, actor_q_two)

            actor_loss = -actor_q_values.mean()

            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_value_(self.actor_net.parameters(), clip_value=1.0) # still no sure about this 0.1
            self.actor_net.optimiser.step()

            # Update target network params
            for target_param, param in zip(self.target_critic_one_net.parameters(), self.critic_one_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_critic_two_net.parameters(), self.critic_two_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
    
    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor_net.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic_one_net.state_dict(), f'models/{filename}_critic.pht')
        torch.save(self.critic_two_net.state_dict(), f'models/{filename}_critic.pht')