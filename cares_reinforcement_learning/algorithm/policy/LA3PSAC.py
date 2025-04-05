"""
Original Paper: https://arxiv.org/abs/2209.00532

https://github.com/h-yamani/RD-PER-baselines/blob/main/LA3P/LA3P/Code/SAC/LA3P_SAC.py
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.LA3PSAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import LA3PSACConfig


class LA3PSAC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: LA3PSACConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)

        self.prioritized_fraction = config.prioritized_fraction

    def _train_critic(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        uniform_sampling: bool,
    ) -> tuple[float, float, float, np.ndarray]:

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards_tensor = rewards_tensor.reshape(len(rewards_tensor), 1)
        dones_tensor = dones_tensor.reshape(len(dones_tensor), 1)

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states_tensor)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states_tensor, next_actions
            )

            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                rewards_tensor * self.reward_scale
                + self.gamma * (1 - dones_tensor) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states_tensor, actions_tensor)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        if uniform_sampling:
            critic_loss_one = hlp.prioritized_approximate_loss(
                td_error_one, self.min_priority, self.per_alpha
            )
            critic_loss_two = hlp.prioritized_approximate_loss(
                td_error_two, self.min_priority, self.per_alpha
            )
            critic_loss_total = critic_loss_one + critic_loss_two
            critic_loss_total /= (
                torch.max(td_error_one, td_error_two)
                .clamp(min=self.min_priority)
                .pow(self.per_alpha)
                .mean()
                .detach()
            )
        else:
            critic_loss_one = hlp.calculate_huber_loss(
                td_error_one, self.min_priority, use_quadratic_smoothing=False
            )
            critic_loss_two = hlp.calculate_huber_loss(
                td_error_two, self.min_priority, use_quadratic_smoothing=False
            )
            critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        return (
            critic_loss_one.item(),
            critic_loss_two.item(),
            critic_loss_total.item(),
            priorities,
        )

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.learn_counter += 1

        uniform_batch_size = int(batch_size * (1 - self.prioritized_fraction))
        priority_batch_size = int(batch_size * self.prioritized_fraction)

        target_update = self.learn_counter % self.target_update_freq == 0

        ######################### UNIFORM SAMPLING #########################
        experiences = memory.sample_uniform(uniform_batch_size)
        states, actions, rewards, next_states, dones, indices = experiences

        info_uniform = {}

        critic_loss_one, critic_loss_two, critic_loss_total, priorities = (
            self._train_critic(
                states,
                actions,
                rewards,
                next_states,
                dones,
                uniform_sampling=True,
            )
        )
        info_uniform["critic_loss_one"] = critic_loss_one
        info_uniform["critic_loss_two"] = critic_loss_two
        info_uniform["critic_loss_total"] = critic_loss_total

        memory.update_priorities(indices, priorities)

        # Train Actor
        weights = np.array([1.0] * len(states))
        weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)

        actor_loss, alpha_loss = self._update_actor_alpha(states_tensor, weights_tensor)
        info_uniform["actor_loss"] = actor_loss
        info_uniform["alpha_loss"] = alpha_loss
        info_uniform["alpha"] = self.alpha.item()

        if target_update:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        ######################### CRITIC PRIORITIZED SAMPLING #########################
        experiences = memory.sample_priority(
            priority_batch_size,
            sampling_stratagy=self.per_sampling_strategy,
            weight_normalisation=self.per_weight_normalisation,
        )
        states, actions, rewards, next_states, dones, indices, _ = experiences

        info_priority = {}

        critic_loss_one, critic_loss_two, critic_loss_total, priorities = (
            self._train_critic(
                states,
                actions,
                rewards,
                next_states,
                dones,
                uniform_sampling=False,
            )
        )
        info_priority["critic_loss_one"] = critic_loss_one
        info_priority["critic_loss_two"] = critic_loss_two
        info_priority["critic_loss_total"] = critic_loss_total

        memory.update_priorities(indices, priorities)

        if target_update:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        ######################### ACTOR PRIORITIZED SAMPLING #########################
        experiences = memory.sample_inverse_priority(priority_batch_size)
        states, _, _, _, _, _, _ = experiences
        weights = np.array([1.0] * len(states))

        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)

        actor_loss, alpha_loss = self._update_actor_alpha(states_tensor, weights_tensor)
        info_priority["actor_loss"] = actor_loss
        info_priority["alpha_loss"] = alpha_loss
        info_priority["alpha"] = self.alpha.item()

        info = {"uniform": info_uniform, "priority": info_priority}

        return info
