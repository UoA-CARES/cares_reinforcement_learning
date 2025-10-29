"""
Original Paper: https://arxiv.org/abs/1812.05905
Code based on: https://github.com/SamsungLabs/tqc_pytorch

This code runs automatic entropy tuning
"""

from typing import Any

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.TQC import Actor, Critic
from cares_reinforcement_learning.util import helpers as hlp
from cares_reinforcement_learning.util.configurations import TQCConfig


class TQC(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: TQCConfig,
        device: torch.device,
    ):
        super().__init__(
            actor_network=actor_network,
            critic_network=critic_network,
            config=config,
            device=device,
        )

        # TQC specific parameters
        self.num_quantiles = config.num_quantiles
        self.num_critics = config.num_critics
        self.kappa = config.kappa

        self.quantiles_total = self.num_quantiles * self.num_critics

        self.top_quantiles_to_drop = config.top_quantiles_to_drop

        self.quantile_taus = torch.FloatTensor(
            [
                i / self.num_quantiles + 0.5 / self.num_quantiles
                for i in range(0, self.num_quantiles)
            ]
        ).to(device)

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                q_value = (
                    self.critic_net(state_tensor, action_tensor)
                    .mean(2)
                    .mean(1, keepdim=True)
                )

        return q_value.item()

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> tuple[dict[str, Any], np.ndarray]:
        batch_size = len(states)
        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
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

        # Compute td_error for PER
        sorted_q_values, _ = torch.sort(q_values.reshape(batch_size, -1))
        top_quantile_q_values = sorted_q_values[
            :, : self.quantiles_total - self.top_quantiles_to_drop
        ]

        td_errors = top_quantile_q_values - q_target
        td_error = td_errors.abs().mean(dim=1)  # mean over quantiles

        critic_loss_total = hlp.calculate_quantile_huber_loss(
            q_values,
            q_target,
            self.quantile_taus,
            kappa=self.kappa,
            use_pairwise_loss=True,
            use_mean_reduction=True,
            use_quadratic_smoothing=True,
        )

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        # Update the Priorities - PER only
        priorities = (
            td_error.clamp(self.min_priority)
            .pow(self.per_alpha)
            .cpu()
            .data.numpy()
            .flatten()
        )

        info = {
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities

    def _update_actor_alpha(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        pi, log_pi, _ = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            mean_qf_pi = self.critic_net(states, pi).mean(2).mean(1, keepdim=True)

        actor_loss = (self.alpha * log_pi - mean_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        alpha_loss = -self.log_alpha * (log_pi + self.target_entropy).detach().mean()

        # update the temperature
        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }
        return info
