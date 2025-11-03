"""
Original Paper: https://arxiv.org/abs/2007.06049
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.LAPSAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import LAPSACConfig


class LAPSAC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: LAPSACConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> tuple[dict[str, Any], np.ndarray]:
        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )

            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                self.reward_scale * rewards + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        huber_lose_one = hlp.calculate_huber_loss(
            td_error_one, self.min_priority, use_quadratic_smoothing=False
        )
        huber_lose_two = hlp.calculate_huber_loss(
            td_error_two, self.min_priority, use_quadratic_smoothing=False
        )
        critic_loss_total = huber_lose_one + huber_lose_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        priorities = (
            torch.max(td_error_one, td_error_two)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        info = {
            "critic_loss_one": huber_lose_one.item(),
            "critic_loss_two": huber_lose_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities
