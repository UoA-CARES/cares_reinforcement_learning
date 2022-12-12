import torch
import numpy as np


class TD3:

    def __init__(self,
                 actor_network,
                 target_actor_network,
                 critic_one,
                 target_critic_one,
                 critic_two,
                 target_critic_two,
                 optimiser,
                 loss,
                 gamma,
                 tau):
        # TODO: check whether each critic needs its parameters
        self.actor_net = actor_network
        self.target_actor_net = target_actor_network

        self.critic_one_net = critic_one
        self.target_critic_one_net = target_critic_one

        self.critic_two_net = critic_two
        self.target_critic_two_net = target_critic_two

        self.optimiser = optimiser
        self.loss = loss

        self.gamma = gamma
        self.tau = tau

        self.learn_counter = 0
        self.policy_update_freq = 2

    def forward(self, observation):
        return self.actor_net.forward(observation)

    def learn(self, experiences):

        batch_size = len(experiences)

        states, actions, rewards, next_states, dones = experiences

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.FloatTensor(np.asarray(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.LongTensor(dones)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():

            next_actions = self.target_actor_net(next_states)

            noise = 0.2 * torch.rand_like(next_actions)
            clipped_noise = noise.clamp(-0.5, 0.5)

            next_actions = torch.clip(next_actions + clipped_noise, torch.FloatTensor(self.act_space.low),
                                      torch.FloatTensor(self.act_space.high))

            target_q_values_one = self.target_critic_one_net(next_states, next_actions)
            target_q_values_two = self.target_critic_two_net(next_states, next_actions)

            target_q_values = torch.min(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one = self.critic_one_net(states, actions)
        q_values_two = self.critic_two_net(states, actions)

        # Update the Critic One
        critic_one_loss = self.loss(q_values_one, q_target)

        self.optimiser.zero_grad()
        critic_one_loss.backward()
        self.optimiser.step()

        # Update Critic Two
        critic_two_loss = self.loss(q_values_two, q_target)

        self.optimiser.zero_grad()
        critic_two_loss.backward()
        self.optimiser.step()

        if self.learn_counter % self.policy_update_freq == 0:

            # Update Actor
            actor_q_one = self.critic_one_net(states, self.actor_net(states))
            actor_q_two = self.critic_two_net(states, self.actor_net(states))

            actor_q_values = torch.min(actor_q_one, actor_q_two)

            actor_loss = -actor_q_values.mean()

            self.optimiser.zero_grad()
            actor_loss.backward()
            self.optimiser.step()

            # Update target network params
            for target_param, param in zip(self.target_critic_one_net.parameters(), self.critic_one_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_critic_two_net.parameters(), self.critic_two_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))
