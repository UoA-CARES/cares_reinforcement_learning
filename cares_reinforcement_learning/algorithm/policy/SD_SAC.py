"""
Original Paper: https://arxiv.org/abs/2209.10081
Code based on: https://github.com/coldsummerday/SD-SAC/blob/main/src/libs/discrete_sac.py

This code runs automatic entropy tuning
"""

from typing import Any

import torch
import torch.nn.functional as F
import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.policy import SACD
from cares_reinforcement_learning.networks.SD_SAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import SD_SACConfig
from cares_reinforcement_learning.util.training_context import TrainingContext

class SD_SAC(SACD):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        env_entropy: float,
        config: SD_SACConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, env_entropy, config, device)
        self.use_clipped_avg_q = config.use_clipped_avg_q
        self.entropy_penalty_beta = config.entropy_penalty_beta
        self.q_clip_epsilon = config.q_clip_epsilon
    

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
        weights: torch.Tensor,
    ) -> float:
        info = {}

        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                _, (action_probs, log_action_probs), _ = self.actor_net(next_states)
            entropies = -torch.sum(action_probs * log_action_probs, dim=-1).squeeze()

            qf1_next_target, qf2_next_target = self.target_critic_net(next_states)
            avg_q_target = torch.mean(torch.stack((qf1_next_target, qf2_next_target), dim=-1), dim=-1)
            q_target = (torch.sum(action_probs * avg_q_target, dim=-1) + self.alpha * entropies) * self.gamma ** self.n_step
            # TODO: Not sure if (1.0 - dones) is correct for n_step calculations?...
            q_target = rewards * self.reward_scale + (1.0 - dones) * q_target.unsqueeze(dim=-1)

        act = actions.long()
        qf1_values, qf2_values = self.critic_net(states)
        qf1_values = qf1_values.gather(1, act)
        qf2_values = qf2_values.gather(1, act)
        qf1_target, qf2_target = self.target_critic_net(states)
        qf1_target = qf1_target.gather(1, act)
        qf2_target = qf2_target.gather(1, act)
        
        clipped_q1 = qf1_target + torch.clamp(qf1_values - qf1_target, -self.q_clip_epsilon, self.q_clip_epsilon)
        q1_loss_1 = F.mse_loss(qf1_values, q_target)
        q1_loss_2 = F.mse_loss(clipped_q1, q_target)
        critic1_loss = torch.maximum(q1_loss_1, q1_loss_2)
        clipq_ratio = torch.mean((q1_loss_2 >= q1_loss_1).float()).item()

        clipped_q2 = qf2_target + torch.clamp(qf2_values - qf2_target, -self.q_clip_epsilon, self.q_clip_epsilon)
        q2_loss_1 = F.mse_loss(qf2_values, q_target)
        q2_loss_2 = F.mse_loss(clipped_q2, q_target)
        critic2_loss = torch.maximum(q2_loss_1, q2_loss_2)
        clipq_ratio = (clipq_ratio + torch.mean((q2_loss_2 >= q2_loss_1).float()).item()) / 2.0

        self.critic_net_optimiser.zero_grad()
        critic_loss_total = critic1_loss + critic2_loss
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        info['critic_loss_one'] = critic1_loss.item()
        info['critic_loss_two'] = critic2_loss.item()
        info['critic_loss_total'] = critic_loss_total.item()
        info['clipped_q1'] = clipped_q1.mean().item()
        info['clipped_q2'] = clipped_q2.mean().item()
        info['clip_ratio'] = clipq_ratio

        return info, {}
    
    def _update_actor_alpha(self, states: torch.Tensor, old_entropies: torch.Tensor) -> tuple[float, float]:
        _, (action_probs, log_action_probs), _ = self.actor_net(states)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                qf1_pi, qf2_pi = self.critic_net(states)

            if self.use_clipped_avg_q:
                q_value = torch.mean(torch.stack((qf1_pi, qf2_pi), dim=-1), dim=-1)
            else:
                q_value = torch.minimum(qf1_pi, qf2_pi)

        entropies = (-action_probs * log_action_probs).sum(dim=-1)
        actor_loss = -(self.alpha * entropies + (action_probs * q_value).sum(dim=-1)).mean()
        entropy_penalty = self.entropy_penalty_beta * F.mse_loss(old_entropies.squeeze(), entropies)
        actor_loss += entropy_penalty

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info = {
            "actor_loss": actor_loss.item(),
            "avg_entropy": entropies.mean().item(),
            "entropy_penalty": entropy_penalty.item(),
        }

        # update the temperature (alpha)
        if self.auto_entropy_tuning:
            alpha_loss = self._update_alpha(entropies)
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()

        return info
    

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        # Use the helper to sample and prepare tensors in one step
        (
            states,
            actions_tensor,
            rewards_tensor,
            next_states,
            dones_tensor,
            extras_tensor,
            weights_tensor,
            _,
        ) = tu.sample_batch_to_tensors(
            memory=memory,
            batch_size=batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        info = {}

        # Update the Critic
        critic_info, priorities = self._update_critic(
            states,
            actions_tensor,
            rewards_tensor,
            next_states,
            dones_tensor,
            weights_tensor,
        )

        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(states, extras_tensor)
            
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        return info