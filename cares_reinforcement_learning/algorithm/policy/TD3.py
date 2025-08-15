"""
Original Paper: https://arxiv.org/abs/1802.09477v3

"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.common import (
    DeterministicPolicy,
    EnsembleCritic,
    TwinQNetwork,
)
from cares_reinforcement_learning.util.configurations import TD3Config


class TD3(VectorAlgorithm):
    def __init__(
        self,
        actor_network: DeterministicPolicy,
        critic_network: TwinQNetwork | EnsembleCritic,
        config: TD3Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_actor_net.eval()  # never in training mode - helps with batch/drop out layers
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        self.gamma = config.gamma
        self.tau = config.tau

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        # Policy noise
        self.min_policy_noise = config.min_policy_noise
        self.policy_noise = config.policy_noise
        self.policy_noise_decay = config.policy_noise_decay

        self.policy_noise_clip = config.policy_noise_clip

        # Action noise
        self.min_action_noise = config.min_action_noise
        self.action_noise = config.action_noise
        self.action_noise_decay = config.action_noise_decay

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq

        self.action_num = self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

    def select_action_from_policy(
        self,
        state: np.ndarray,
        evaluation: bool = False,
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(
                    0, scale=self.action_noise, size=self.action_num
                )
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor_net.train()

        return action

    def _calculate_value(self, state: np.ndarray, action: np.ndarray) -> float:  # type: ignore[override]
        state_tensor = torch.FloatTensor(state).to(self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.FloatTensor(action).to(self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                q_values_one, q_values_two = self.critic_net(
                    state_tensor, action_tensor
                )
                q_value = torch.minimum(q_values_one, q_values_two)

        return q_value[0].item()

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

        critic_loss_one = F.mse_loss(q_values_one, q_target, reduction="none")
        critic_loss_one = (critic_loss_one * weights).mean()

        critic_loss_two = F.mse_loss(q_values_two, q_target, reduction="none")
        critic_loss_two = (critic_loss_two * weights).mean()

        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
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

    # Weights is set for methods like MAPERTD3 that use weights in the actor update
    def _update_actor(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        actions = self.actor_net(states)

        with hlp.evaluating(self.critic_net):
            actor_q_values, _ = self.critic_net(states, actions)

        actor_loss = -actor_q_values.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        actor_info = {
            "actor_loss": actor_loss.item(),
        }

        return actor_info

    # TODO use training_step with decay rates
    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        self.learn_counter += 1

        self.policy_noise *= self.policy_noise_decay
        self.policy_noise = max(self.min_policy_noise, self.policy_noise)

        self.action_noise *= self.action_noise_decay
        self.action_noise = max(self.min_action_noise, self.action_noise)

        if self.use_per_buffer:
            experiences = memory.sample_priority(
                batch_size,
                sampling_stratagy=self.per_sampling_strategy,
                weight_normalisation=self.per_weight_normalisation,
            )
            states, actions, rewards, next_states, dones, indices, weights = experiences
        else:
            experiences = memory.sample_uniform(batch_size)
            states, actions, rewards, next_states, dones, _ = experiences
            weights = [1.0] * batch_size

        batch_size = len(states)

        # Convert into tensor
        states_tensor = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states_tensor = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)
        weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)

        # Reshape to batch_size
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)
        weights_tensor = weights_tensor.reshape(batch_size, 1)

        info: dict[str, Any] = {}

        # Update the Critic
        critic_info, priorities = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
        )
        info |= critic_info

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_info = self._update_actor(states_tensor, weights_tensor)
            info |= actor_info

            # Update target network params
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
            hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        # Update the Priorities
        if self.use_per_buffer:
            memory.update_priorities(indices, priorities)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht", map_location=torch.device('cpu')))
        self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht", map_location=torch.device('cpu')))
        logging.info("models has been loaded...")
