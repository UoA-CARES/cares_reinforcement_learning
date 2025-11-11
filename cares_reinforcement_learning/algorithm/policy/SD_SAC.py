"""
Original Paper: https://arxiv.org/abs/2209.10081
Code based on: https://github.com/coldsummerday/SD-SAC

This code runs automatic entropy tuning
"""

import torch
import torch.nn.functional as F
import cares_reinforcement_learning.util.helpers as hlp
from SACD import (
    SACD,
    Actor,
    Critic,
    SACDConfig,
)

class SD_SAC(SACD):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SACDConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)

        self.clip_q_epsilon = config.clip_q_epsilon

    def get_action_entropy(self, state) -> torch.Tensor:
        _, (action_probs, log_action_probs), _ = self.actor_net(state)
        entropy = -torch.sum(action_probs * log_action_probs, dim=-1)
        return entropy
    
    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
        entropies: torch.Tensor,
    ) -> float:
        info = {}

        with torch.no_grad():
            qf1_next_target, qf2_next_target = self.target_critic_net(next_states)
            _, (action_probs, _) = self.actor_net(next_states)
            
            avg_q_target = torch.mean(torch.stack((qf1_next_target, qf2_next_target), dim=-1), dim=-1)
            q_target = action_probs * (rewards * self.reward_scale + (1.0 - dones) * avg_q_target * self.gamma) + self.alpha * entropies

            qf1_values, qf2_values = self.critic_net(states)
            qf1_target, qf2_target = self.target_critic_net(states)
        
        clipped_q1 = qf1_target + torch.clamp(qf1_values - qf1_target, -self.clip_q_epsilon, self.clip_q_epsilon)
        q1_loss_1 = F.mse_loss(qf1_values, q_target) * weights
        q1_loss_2 = F.mse_loss(clipped_q1, q_target) * weights
        critic1_loss = torch.maximum(q1_loss_1, q1_loss_2)
        clipq_ratio = torch.mean((q1_loss_2 >= q1_loss_1).float()).item()
        info['clipped_q1'] = clipped_q1.mean().item()

        clipped_q2 = qf2_target + torch.clamp(qf2_values - qf2_target, -self.clip_q_epsilon, self.clip_q_epsilon)
        q2_loss_1 = F.mse_loss(qf2_values, q_target) * weights
        q2_loss_2 = F.mse_loss(clipped_q2, q_target) * weights
        critic2_loss = torch.maximum(q2_loss_1, q2_loss_2)
        clipq_ratio = (clipq_ratio + torch.mean((q2_loss_2 >= q2_loss_1).float()).item()) / 2.0
        info['clipped_q2'] = clipped_q2.mean().item()
        info['clip_ratio'] = clipq_ratio

        self.critic_net_optimiser.zero_grad()
        critic_loss_total = critic1_loss + critic2_loss
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        info['critic_loss_one'] = critic1_loss.item()
        info['critic_loss_two'] = critic2_loss.item()
        info['critic_loss_total'] = critic_loss_total.item()

        return info, {}