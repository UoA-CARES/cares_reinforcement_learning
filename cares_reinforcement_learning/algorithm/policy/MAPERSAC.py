"""
Original Paper: https://openreview.net/pdf?id=WuEiafqdy9H
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


class MAPERSAC:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        gamma: float,
        tau: float,
        per_alpha: float,
        min_priority: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        device: torch.device,
    ):

        self.type = "policy"
        self.device = device

        self.actor_net = actor_network.to(self.device)

        self.critic_net = critic_network.to(self.device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)

        self.gamma = gamma
        self.tau = tau

        self.per_alpha = per_alpha
        self.min_priority = min_priority

        self.learn_counter = 0
        self.policy_update_freq = 1

        self.target_entropy = -action_num

        # MAPER-PER parameters
        self.scale_r = 1.0
        self.scale_s = 1.0

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.actor_net_optimiser = optim.Adam(self.actor_net.parameters(), lr=actor_lr)

        self.critic_net_optimiser = optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        # Set to initial alpha to 1.0 according to other baselines.
        init_temperature = 1.0
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=1e-3)

    def _split_output(self, target):
        return target[:, 0], target[:, 1], target[:, 2:]

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            if evaluation is False:
                (action, _, _) = self.actor_net(state_tensor)
            else:
                (_, _, action) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> float:
        return self.log_alpha.exp()

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        # Sample replay buffer
        experiences = memory.sample_priority(batch_size)
        states, actions, rewards, next_states, dones, indices, weights = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)
        weights = torch.LongTensor(np.asarray(weights)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # Get current Q estimates
        output_one, output_two = self.critic_net(states.detach(), actions.detach())
        q_value_one, predicted_reward_one, next_states_one = self._split_output(
            output_one
        )
        q_value_two, predicted_reward_two, next_states_two = self._split_output(
            output_two
        )

        diff_reward_one = 0.5 * torch.pow(
            predicted_reward_one.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0
        ).reshape(-1, 1)
        diff_reward_two = 0.5 * torch.pow(
            predicted_reward_two.reshape(-1, 1) - rewards.reshape(-1, 1), 2.0
        ).reshape(-1, 1)

        diff_next_states_one = 0.5 * torch.mean(
            torch.pow(
                next_states_one - next_states,
                2.0,
            ),
            -1,
        )
        diff_next_states_one = diff_next_states_one.reshape(-1, 1)

        diff_next_states_two = 0.5 * torch.mean(
            torch.pow(
                next_states_two - next_states,
                2.0,
            ),
            -1,
        )
        diff_next_states_two = diff_next_states_two.reshape(-1, 1)

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)
            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            next_values_one, _, _ = self._split_output(target_q_values_one)
            next_values_two, _, _ = self._split_output(target_q_values_two)
            min_next_target = torch.minimum(next_values_one, next_values_two).reshape(
                -1, 1
            )
            target_q_values = min_next_target - self.alpha * next_log_pi

            predicted_rewards = (
                (
                    predicted_reward_one.reshape(-1, 1)
                    + predicted_reward_two.reshape(-1, 1)
                )
                / 2
            ).reshape(-1, 1)

            q_target = predicted_rewards + self.gamma * (1 - dones) * target_q_values

        diff_td_one = F.mse_loss(q_value_one.reshape(-1, 1), q_target, reduction="none")
        diff_td_two = F.mse_loss(q_value_two.reshape(-1, 1), q_target, reduction="none")

        critic_one_loss = (
            diff_td_one
            + self.scale_r * diff_reward_one
            + self.scale_s * diff_next_states_one
        )

        critic_two_loss = (
            diff_td_two
            + self.scale_r * diff_reward_two
            + self.scale_s * diff_next_states_two
        )

        critic_loss_total = (critic_one_loss * weights).mean() + (
            critic_two_loss * weights
        ).mean()

        # train critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # calculate priority
        diff_td_mean = torch.cat([diff_td_one, diff_td_two], -1)
        diff_td_mean = torch.mean(diff_td_mean, 1)
        diff_td_mean = diff_td_mean.reshape(-1, 1)
        diff_td_mean = diff_td_mean[:, 0].detach().data.cpu().numpy()

        diff_reward_mean = torch.cat([diff_reward_one, diff_reward_two], -1)
        diff_reward_mean = torch.mean(diff_reward_mean, 1)
        diff_reward_mean = diff_reward_mean.reshape(-1, 1)
        diff_reward_mean = diff_reward_mean[:, 0].detach().data.cpu().numpy()

        diff_next_state_mean = torch.cat(
            [diff_next_states_one, diff_next_states_two], -1
        )
        diff_next_state_mean = torch.mean(diff_next_state_mean, 1)
        diff_next_state_mean = diff_next_state_mean.reshape(-1, 1)
        diff_next_state_mean = diff_next_state_mean[:, 0].detach().data.cpu().numpy()

        # calculate priority
        priorities = np.array(
            [
                (
                    diff_td_mean
                    + self.scale_s * diff_next_state_mean
                    + self.scale_r * diff_reward_mean
                )
            ]
        ).reshape(-1)

        priorities.clip(min=self.min_priority)
        priorities = priorities**self.per_alpha

        pi, log_pi, _ = self.actor_net(states)
        qf1_pi, qf2_pi = self.critic_net(states, pi)
        qf_pi_one, _, _ = self._split_output(qf1_pi)
        qf_pi_two, _, _ = self._split_output(qf2_pi)
        min_qf_pi = torch.minimum(qf_pi_one, qf_pi_two)

        actor_loss = torch.mean(
            (torch.exp(self.log_alpha).detach() * log_pi - min_qf_pi) * weights
        )

        # Update the Actor
        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # Update the temperature
        alpha_loss = (
            -weights * self.log_alpha * (log_pi + self.target_entropy).detach()
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

        # Update Scales
        if self.learn_counter == 1:
            self.scale_r = np.mean(diff_td_mean) / (np.mean(diff_next_state_mean))
            self.scale_s = np.mean(diff_td_mean) / (np.mean(diff_next_state_mean))

        memory.update_priorities(indices, priorities)

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
