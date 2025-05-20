"""
Original Paper: https://openreview.net/pdf?id=WuEiafqdy9H

https://github.com/h-yamani/RD-PER-baselines/blob/main/MAPER/MfRL_Cont/algorithms/sac/masac.py
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.MAPERSAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import MAPERSACConfig


class MAPERSAC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: MAPERSACConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)

        # MAPER-PER parameters
        self.scale_r = 1.0
        self.scale_s = 1.0

    def _split_output(
        self, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return target[:, 0], target[:, 1], target[:, 2:]

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                output_one, output_two = self.critic_net(state_tensor, action_tensor)

                q_value_one, _, _ = self._split_output(output_one)
                q_value_two, _, _ = self._split_output(output_two)

                q_value = torch.minimum(q_value_one, q_value_two)

        return q_value.item()

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
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
            with hlp.evaluating(self.actor_net):
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

        critic_loss_one = (
            diff_td_one
            + self.scale_r * diff_reward_one
            + self.scale_s * diff_next_states_one
        )
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = (
            diff_td_two
            + self.scale_r * diff_reward_two
            + self.scale_s * diff_next_states_two
        )
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # calculate priority
        diff_td_mean = torch.cat([diff_td_one, diff_td_two], -1)
        diff_td_mean = torch.mean(diff_td_mean, 1)
        diff_td_mean = diff_td_mean.reshape(-1, 1)
        numpy_td_mean = diff_td_mean[:, 0].detach().data.cpu().numpy()

        diff_reward_mean = torch.cat([diff_reward_one, diff_reward_two], -1)
        diff_reward_mean = torch.mean(diff_reward_mean, 1)
        diff_reward_mean = diff_reward_mean.reshape(-1, 1)
        diff_reward_mean_numpy = diff_reward_mean[:, 0].detach().data.cpu().numpy()

        diff_next_state_mean = torch.cat(
            [diff_next_states_one, diff_next_states_two], -1
        )
        diff_next_state_mean = torch.mean(diff_next_state_mean, 1)
        diff_next_state_mean = diff_next_state_mean.reshape(-1, 1)
        diff_next_state_mean_numpy = (
            diff_next_state_mean[:, 0].detach().data.cpu().numpy()
        )

        # calculate priority
        priorities = (
            numpy_td_mean
            + self.scale_s * diff_next_state_mean_numpy
            + self.scale_r * diff_reward_mean_numpy
        )
        priorities_tensor = torch.Tensor(priorities)

        priorities = (
            priorities_tensor.clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        # Update Scales
        if self.learn_counter == 1:
            self.scale_r = np.mean(numpy_td_mean) / (
                np.mean(diff_next_state_mean_numpy)
            )
            self.scale_s = np.mean(numpy_td_mean) / (
                np.mean(diff_next_state_mean_numpy)
            )

        info = {
            "critic_loss_one": critic_loss_one.item(),
            "critic_loss_two": critic_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities

    def _update_actor_alpha(
        self, states: torch.Tensor, weights: torch.Tensor
    ) -> dict[str, Any]:
        pi, log_pi, _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            output_one, output_two = self.critic_net(states, pi)

        qf_pi_one, _, _ = self._split_output(output_one)
        qf_pi_two, _, _ = self._split_output(output_two)
        min_qf_pi = torch.minimum(qf_pi_one, qf_pi_two)

        actor_loss = torch.mean(
            (torch.exp(self.log_alpha).detach() * log_pi - min_qf_pi) * weights
        )

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

        info = {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }

        return info
