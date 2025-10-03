"""
Original Paper: https://arxiv.org/abs/2209.00532

https://github.com/h-yamani/RD-PER-baselines/blob/main/LA3P/LA3P/Code/SAC/LA3P_SAC.py
"""

from typing import Any

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.policy import SAC
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.LA3PSAC import Actor, Critic
from cares_reinforcement_learning.util.configurations import LA3PSACConfig
from cares_reinforcement_learning.util.training_context import TrainingContext


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

    # pylint: disable-next=arguments-differ, arguments-renamed
    def _update_critic(  # type: ignore[override]
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        uniform_sampling: bool,
    ) -> tuple[dict[str, Any], np.ndarray]:

        # Convert into tensors using helper method
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            _,
        ) = tu.batch_to_tensors(
            states,
            actions,
            rewards,
            next_states,
            dones,
            self.device,
        )

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

        info = {
            "critic_loss_one": critic_loss_one.item(),
            "critic_loss_two": critic_loss_two.item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        uniform_batch_size = int(batch_size * (1 - self.prioritized_fraction))
        priority_batch_size = int(batch_size * self.prioritized_fraction)

        target_update = self.learn_counter % self.target_update_freq == 0

        ######################### UNIFORM SAMPLING #########################
        experiences = memory.sample_uniform(uniform_batch_size)
        states, actions, rewards, next_states, dones, indices = experiences

        info_uniform: dict[str, Any] = {}

        critic_info, priorities = self._update_critic(
            states,
            actions,
            rewards,
            next_states,
            dones,
            uniform_sampling=True,
        )
        info_uniform |= critic_info

        memory.update_priorities(indices, priorities)

        # Train Actor
        weights = np.array([1.0] * len(states))
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)

        actor_info = self._update_actor_alpha(states_tensor, weights_tensor)
        info_uniform |= actor_info
        info_uniform["alpha"] = self.alpha.item()

        if target_update:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        ######################### CRITIC PRIORITIZED SAMPLING #########################
        experiences = memory.sample_priority(
            priority_batch_size,
            sampling_strategy=self.per_sampling_strategy,
            weight_normalisation=self.per_weight_normalisation,
        )
        states, actions, rewards, next_states, dones, indices, _ = experiences

        info_priority: dict[str, Any] = {}

        critic_info, priorities = self._update_critic(
            states,
            actions,
            rewards,
            next_states,
            dones,
            uniform_sampling=False,
        )
        info_priority |= critic_info

        memory.update_priorities(indices, priorities)

        if target_update:
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)

        ######################### ACTOR PRIORITIZED SAMPLING #########################
        experiences = memory.sample_inverse_priority(priority_batch_size)
        states, _, _, _, _, _, _ = experiences
        weights = np.array([1.0] * len(states))

        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        weights_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)

        actor_info = self._update_actor_alpha(states_tensor, weights_tensor)
        info_priority |= actor_info
        info_priority["alpha"] = self.alpha.item()

        info = {"uniform": info_uniform, "priority": info_priority}

        return info
