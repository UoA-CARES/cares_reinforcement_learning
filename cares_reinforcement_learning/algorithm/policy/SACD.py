"""
Original Paper: https://arxiv.org/pdf/1910.07207
Code based on: https://github.com/p-christ/Deep-Reinforcement-Learning-Algorithms-with-PyTorch/blob/master/agents/actor_critic_agents/SAC_Discrete.py

This code runs automatic entropy tuning
"""

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.networks.SACD import Actor, Critic
from cares_reinforcement_learning.util.configurations import SACDConfig
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class SACD(SAC):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        config: SACDConfig,
        device: torch.device,
    ):
        super().__init__(actor_network, critic_network, config, device)
        self.policy_type = "discrete_policy"
        self.action_num = self.actor_net.num_actions
        self.target_entropy = (np.log(self.action_num) * config.target_entropy_multiplier)


    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
    

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:

        self.actor_net.eval()

        state = action_context.state
        evaluation = action_context.evaluation

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)
            if evaluation:
                (_, _, action) = self.actor_net(state_tensor)
            else:
                (action, _, _) = self.actor_net(state_tensor)
            action = action.cpu().numpy().flatten()
        self.actor_net.train()
        return action


    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> float:
        # Make sure we are not training target networks - Would gradients propagate into targets?
        with torch.no_grad():
            with hlp.evaluating(self.actor_net):
                _, (action_probs, log_actions_probs), _ = self.actor_net(next_states)

            qf1_next_target, qf2_next_target = self.target_critic_net(next_states)

            min_qf_next_target = action_probs * (
                torch.minimum(qf1_next_target, qf2_next_target)
                - self.alpha * log_actions_probs
            )

            min_qf_next_target = min_qf_next_target.sum(dim=1).unsqueeze(-1)
            next_q_value = (
                rewards * self.reward_scale
                + (1.0 - dones) * min_qf_next_target * self.gamma ** self.n_step
            ).flatten()

        # Get the q_value of the action taken
        act = torch.as_tensor(actions[:, np.newaxis], device="cuda", dtype=torch.long).flatten(1)
        q_values_one, q_values_two = self.critic_net(states)
        q_values_one = q_values_one.gather(1, act).flatten()
        q_values_two = q_values_two.gather(1, act).flatten()

        critic_loss_one = F.mse_loss(q_values_one, next_q_value)
        critic_loss_two = F.mse_loss(q_values_two, next_q_value)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        td_error_one = (q_values_one - next_q_value).abs()
        td_error_two = (q_values_two - next_q_value).abs()

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


    def _update_actor_alpha(self, states: torch.Tensor) -> tuple[float, float]:
        _, (action_probs, log_action_probs), _ = self.actor_net(states)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                qf1_pi, qf2_pi = self.critic_net(states)
            min_qf_pi = torch.minimum(qf1_pi, qf2_pi)


        entropy = (- action_probs * log_action_probs)
        actor_loss = - (self.alpha.detach() * entropy + action_probs * min_qf_pi).sum(dim=1).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info = {
            "actor_loss": actor_loss.item(),
            "avg_entropy": entropy.sum(dim=1).mean().item(),
        }

        # update the temperature (alpha)
        if self.auto_entropy_tuning:
            alpha_loss = self._update_alpha(entropy)
            info["alpha_loss"] = alpha_loss.item()
            info["alpha"] = self.alpha.item()

        return info
    

    def _update_alpha(self, entropy: torch.Tensor) -> torch.Tensor:
        # update the temperature (alpha)
        log_prob = -entropy.detach().sum(dim=1) + self.target_entropy
        alpha_loss = -(self.log_alpha * log_prob).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return alpha_loss


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
            _,
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
        )

        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor_alpha(states)
            
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        return info


    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        return 0.0