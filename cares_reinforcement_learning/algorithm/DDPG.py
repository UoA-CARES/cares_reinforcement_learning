
import os
import copy
import logging
import numpy as np
import torch
import torch.nn.functional as F


class DDPG:

    def __init__(self,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 device
                 ):

        self.type = "policy"
        self.actor_net  = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net  = copy.deepcopy(self.actor_net).to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau

        self.device = device

    def select_action_from_policy(self, state, evaluation=None):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        return action

    def train_policy(self, experiences):
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

        # We do not want the gradients calculated for any of the target old_networks, we manually update the parameters
        with torch.no_grad():
            next_actions    = self.target_actor_net(next_states)
            target_q_values = self.target_critic_net(next_states, next_actions)
            q_target        = rewards + self.gamma * (1 - dones) * target_q_values

        q_values = self.critic_net(states, actions)

        # Update the Critic Network
        critic_loss = F.mse_loss(q_values, q_target)

        self.critic_net.optimiser.zero_grad()
        critic_loss.backward()
        self.critic_net.optimiser.step()

        # Update the Actor Network
        actor_q    = self.critic_net(states, self.actor_net(states))
        actor_loss = -actor_q.mean()

        self.actor_net.optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net.optimiser.step()


        for target_param, param in zip(self.target_critic_net.parameters(), self.critic_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.target_actor_net.parameters(), self.actor_net.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))


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
