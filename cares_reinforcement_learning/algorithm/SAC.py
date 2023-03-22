
import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F

class SAC:
    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 device):

        self.actor_net  = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.device = device

        self.target_entropy = -np.prod(action_num)

        init_temperature = 0.01
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3, betas=(0.9, 0.999))

    def select_action_from_policy(self, state):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            mu, _, _, _ = self.actor_net(state_tensor, compute_pi=False, compute_log_pi=False)
            mu = mu.cpu().data.numpy().flatten()
        return mu

    def train_policy(self, experiences):
        self.learn_counter += 1

        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(next_states, actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two) - self.alpha.detach() * log_pi

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_1 = F.mse_loss(q_values_one, q_target)
        critic_loss_2 = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        # Update the Critic
        self.critic_net.optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net.optimiser.step()

        if self.learn_counter % self.policy_update_freq == 0:
            _, pi, log_pi, log_std = self.actor_net(states)
            actor_q_one, actor_q_two = self.critic_net(states, pi)
            actor_q_values = torch.minimum(actor_q_one, actor_q_two)

            actor_loss = (self.alpha.detach() * log_pi - actor_q_values).mean()

            self.actor_net.optimiser.zero_grad()
            actor_loss.backward()
            self.actor_net.optimiser.step()

            self.log_alpha_optimizer.zero_grad()
            alpha_loss = (self.alpha * (-log_pi - self.target_entropy).detach()).mean()
            alpha_loss.backward()
            self.log_alpha_optimizer.step()

            for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


    @property
    def alpha(self):
        return self.log_alpha.exp()

    def save_models(self, filename):
        dir_exists = os.path.exists("models")

        if not dir_exists:
            os.makedirs("models")
        torch.save(self.actor_net.state_dict(),  f'models/{filename}_actor.pht')
        torch.save(self.critic_net.state_dict(), f'models/{filename}_critic.pht')
        logging.info("models has been loaded...")

    def load_models(self, filename):
        self.actor_net.load_state_dict(torch.load(f'models/{filename}_actor.pht'))
        self.critic_net.load_state_dict(torch.load(f'models/{filename}_critic.pht'))
        logging.info("models has been loaded...")






