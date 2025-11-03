"""
Original Paper: https://arxiv.org/abs/2007.06049
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import TD3
from cares_reinforcement_learning.networks.PALTD3 import Actor, Critic
from cares_reinforcement_learning.util.configurations import PALTD3Config


class PALTD3(TD3):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: PALTD3Config,
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
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        pal_loss_one = hlp.prioritized_approximate_loss(
            td_error_one, self.min_priority, self.per_alpha
        )
        pal_loss_two = hlp.prioritized_approximate_loss(
            td_error_two, self.min_priority, self.per_alpha
        )
        critic_loss_total = pal_loss_one + pal_loss_two

        critic_loss_total /= (
            torch.max(td_error_one, td_error_two)
            .clamp(min=self.min_priority)
            .pow(self.per_alpha)
            .mean()
            .detach()
        )

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        batch_size = states.shape[0]
        priorities = np.array([1.0] * batch_size)

        info = {
            "critic_loss_one": pal_loss_one.item(),
            "critic_loss_two": pal_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities
