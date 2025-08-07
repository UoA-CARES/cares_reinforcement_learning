"""
Original Paper: https://arxiv.org/pdf/1902.05605
Code based on: https://github.com/modelbased/minirllab/blob/main/agents/sac_crossq.py

This code runs automatic entropy tuning
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.CrossQ import Actor, Critic
from cares_reinforcement_learning.util.configurations import CrossQConfig


class CrossQ(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: CrossQConfig,
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

        cat_states = torch.cat([states, next_states], dim=0)
        cat_actions = torch.cat([actions, next_actions], dim=0)

        cat_q_values_one, cat_q_values_two = self.critic_net(cat_states, cat_actions)

        q_values_one, q_values_one_next = torch.chunk(cat_q_values_one, chunks=2, dim=0)
        q_values_two, q_values_two_next = torch.chunk(cat_q_values_two, chunks=2, dim=0)

        target_q_values = (
            torch.minimum(q_values_one_next, q_values_two_next)
            - self.alpha * next_log_pi
        )

        q_target = (
            rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
        )
        torch.detach(q_target)

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            torch.max(td_error_one, td_error_two)
            .clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        info = {
            "critic_loss_one": critic_loss_one.item(),
            "critic_loss_two": critic_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities
