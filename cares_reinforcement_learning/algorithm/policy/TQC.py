"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/pranz24/pytorch-soft-actor-critic/blob/master/sac.py.

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.util import helpers as hlp


class TQC:
    def __init__(
        self,
        actor_network,
        critic_network,
        gamma,
        tau,
        top_quantiles_to_drop,
        action_num,
        actor_lr,
        critic_lr,
        alpha_lr,
        device,
    ):
        self.type = "policy"

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.gamma = gamma
        self.tau = tau
        self.top_quantiles_to_drop = top_quantiles_to_drop

        self.quantiles_total = (
            self.critic_net.num_quantiles * self.critic_net.num_critics
        )

        self.learn_counter = 0
        self.policy_update_freq = 1

        self.device = device

        self.target_entropy = -action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # pylint: disable-next=unused-argument
    def select_action_from_policy(self, state, evaluation=False, noise_scale=0):
        # note that when evaluating this algorithm we need to select tanh(mean) as action
        # so _, _, action = self.actor_net(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (
                    action,
                    _,
                    _,
                ) = self.actor_net(state_tensor)
            else:
                (
                    _,
                    _,
                    action,
                ) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train_policy(self, memory, batch_size):
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)

            # compute and cut quantiles at the next state
            # batch x nets x quantiles
            target_q_values = self.target_critic_net(next_states, next_actions)
            sorted_target_q_values, _ = torch.sort(
                target_q_values.reshape(batch_size, -1)
            )
            top_quantile_target_q_values = sorted_target_q_values[
                :, : self.quantiles_total - self.top_quantiles_to_drop
            ]

            # compute target
            q_target = rewards + (1 - dones) * self.gamma * (
                top_quantile_target_q_values - self.alpha * next_log_pi
            )

        q_values = self.critic_net(states, actions)
        critic_loss_total = hlp.quantile_huber_loss_f(q_values, q_target)

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # --- Policy and alpha loss ---
        new_action, log_pi, _ = self.actor_net(states)

        mean_qf_pi = self.critic_net(states, new_action).mean(2).mean(1, keepdim=True)
        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.learn_counter % self.policy_update_freq == 0:
            for target_param, param in zip(
                self.target_critic_net.parameters(), self.critic_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

    def save_models(self, filename, filepath="models"):
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
