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
import cares_reinforcement_learning.util.training_utils as tu
from cares_reinforcement_learning.algorithm.algorithm import VectorAlgorithm
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.common import (
    DeterministicPolicy,
    EnsembleCritic,
    TwinQNetwork,
)
from cares_reinforcement_learning.util.configurations import TD3Config
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


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

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        self.actor_net.eval()

        state = action_context.state
        evaluation = action_context.evaluation

        assert isinstance(state, np.ndarray)

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
            with hlp.evaluating(self.actor_net):
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

    def update_networks(
        self,
        memory: MemoryBuffer,
        indices: np.ndarray,
        states_tensor: torch.Tensor,
        actions_tensor: torch.Tensor,
        rewards_tensor: torch.Tensor,
        next_states_tensor: torch.Tensor,
        dones_tensor: torch.Tensor,
        weights_tensor: torch.Tensor,
    ) -> dict[str, Any]:

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
            # Update the Actor and Alpha
            actor_info = self._update_actor(states_tensor, weights_tensor)
            info |= actor_info

            # Update target network params
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
            hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

        # Update the Priorities
        if self.use_per_buffer:
            memory.update_priorities(indices, priorities)

        return info

    # TODO use training_step with decay rates
    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        # TODO replace with training_step based approach to avoid having to save this value
        self.policy_noise *= self.policy_noise_decay
        self.policy_noise = max(self.min_policy_noise, self.policy_noise)

        # TODO replace with training_step based approach to avoid having to save this value
        self.action_noise *= self.action_noise_decay
        self.action_noise = max(self.min_action_noise, self.action_noise)

        # Use the helper to sample and prepare tensors in one step
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
            indices,
        ) = tu.sample_batch_to_tensors(
            memory=memory,
            batch_size=batch_size,
            device=self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        info = self.update_networks(
            memory,
            indices,
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
        )

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_actor": self.target_actor_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
            "policy_noise": self.policy_noise,
            "action_noise": self.action_noise,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.target_actor_net.load_state_dict(checkpoint["target_actor"])

        self.critic_net.load_state_dict(checkpoint["critic"])
        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])

        self.learn_counter = checkpoint.get("learn_counter", 0)

        self.policy_noise = checkpoint.get("policy_noise", self.policy_noise)
        self.action_noise = checkpoint.get("action_noise", self.action_noise)

        logging.info("models, optimisers, and training state have been loaded...")
