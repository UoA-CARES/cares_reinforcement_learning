"""
Original Paper: https://proceedings.neurips.cc/paper_files/paper/2018/file/f02208a057804ee16ac72ff4d3cec53b-Paper.pdf
Code based on: Self-implemented.

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F


class MBRL_STEVE_SAC:
    """
    Soft Actor Critic: Adding a entropy of policy term into the objective
    function for automatic exploration. This agent maximize the reward will
    maximizing the exploration.

    """

    def __init__(
        self,
        actor_network,
        critic_network,
        world_network,
        gamma,
        tau,
        action_num,
        actor_lr,
        critic_lr,
        use_bounded_active,
        horizon,
        device,
    ):
        self.device = device
        self.batch_size = None
        self.use_bounded_active = use_bounded_active
        self.horizon = horizon

        self.type = "mbrl"
        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)
        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.gamma = gamma
        self.tau = tau
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
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha])
        # World model
        self.world_model = world_network
        self.world_model.to(device)

    # pylint: disable-next=unused-argument to keep the same interface
    def select_action_from_policy(self, state, evaluation=False, noise_scale=0):
        """
        Select a action for executing. It is the only channel that an agent
        will communicate the the actual environment.

        """
        # note that when evaluating this algorithm we need to select mu as
        # action so _, _, action = self.actor_net.sample(state_tensor)
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net.sample(state_tensor)
            else:
                (_, _, action) = self.actor_net.sample(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self):
        """
        A variatble decide to what extend entropy shoud be valued.
        """
        return self.log_alpha.exp()

    def train_policy(self, experiences):
        """
        Train the policy with Model-Based Value Expansion. A family of MBRL.

        """
        self.learn_counter += 1
        info = {}
        ### Standarize the data.
        (
            states,
            actions,
            rewards,
            next_states,
            dones,
            _,
            next_actions,
            next_rewards,
        ) = experiences
        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_rewards = torch.FloatTensor(np.asarray(next_rewards)).to(self.device)
        next_actions = torch.FloatTensor(np.asarray(next_actions)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        next_rewards = next_rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)
        not_dones = 1 - dones

        assert len(states.shape) >= 2
        assert len(actions.shape) == 2
        assert len(rewards.shape) == 2 and rewards.shape[1] == 1
        assert len(next_states.shape) >= 2
        assert len(not_dones.shape) == 2 and not_dones.shape[1] == 1
        self.batch_size = states.shape[0]
        self.train_world_model(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            next_actions=next_actions,
            next_rewards=next_rewards,
        )

        with torch.no_grad():
            q_target = torch.sum(all_vars * all_means, dim=0)

        q_values_one, q_values_two = self.critic_net(states, actions)
        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two
        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        ##################     Update the Actor Second     #####################
        pi, first_log_p, _ = self.actor_net.sample(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * first_log_p) - min_qf_pi).mean()

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # update the temperature
        alpha_loss = -(
            self.log_alpha * (first_log_p + self.target_entropy).detach()
        ).mean()
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

        info["q_target"] = q_target
        info["q_values_one"] = q_values_one
        info["q_values_two"] = q_values_two
        info["q_values_min"] = torch.minimum(q_values_one, q_values_two)
        info["critic_loss_total"] = critic_loss_total
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["actor_loss"] = actor_loss
        return info

    def train_world_model(
        self, states, actions, rewards, next_states, next_actions, next_rewards
    ):
        """
        Train the world model with sampled experiences.

        :param (Dictionary) statistics -- The mean and var of collected
        transitions.
        :param (Tensor) transitions -- The data used for training.
        """
        # mask the nones and zeros out.
        ok_masks = []
        for i in range(len(states)):
            if torch.sum(next_actions[i]) == 0 or next_rewards[i] == np.inf:
                ok_masks.append(False)
            else:
                ok_masks.append(True)
        states = states[ok_masks]
        actions = actions[ok_masks]
        rewards = rewards[ok_masks]
        next_states = next_states[ok_masks]
        next_actions = next_actions[ok_masks]
        next_rewards = next_rewards[ok_masks]
        self.world_model.train_world(
            states, actions, rewards, next_states, next_actions, next_rewards
        )

    def save_models(self, filename, filepath="models"):
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)
        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pth")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pth")
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pth"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pth"))
        logging.info("models has been loaded...")