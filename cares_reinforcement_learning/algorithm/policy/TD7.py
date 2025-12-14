"""
Original Paper: https://arxiv.org/pdf/2306.02451

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
from cares_reinforcement_learning.networks.TD7 import Actor, Critic, Encoder
from cares_reinforcement_learning.util.configurations import TD7Config
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class TD7(VectorAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        encoder_network: Encoder,
        config: TD7Config,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(device)
        self.critic_net = critic_network.to(device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_actor_net.eval()  # never in training mode - helps with batch/drop out layers
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        self.encoder_net = encoder_network.to(device)
        self.fixed_encoder_net = copy.deepcopy(self.encoder_net).to(self.device)
        self.target_fixed_encoder_net = copy.deepcopy(self.encoder_net).to(self.device)

        self.checkpoint_actor = copy.deepcopy(self.actor_net).to(self.device)
        self.checkpoint_encoder = copy.deepcopy(self.encoder_net).to(self.device)

        self.gamma = config.gamma
        self.tau = config.tau

        self.target_update_freq = config.target_update_rate

        # Checkpoint tracking
        self.max_eps_checkpointing = config.max_eps_checkpointing
        self.steps_before_checkpointing = config.steps_before_checkpointing
        self.reset_weight = config.reset_weight

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.max_eps_before_update = 1
        self.min_return = 1e8
        self.best_min_return = -1e8

        # Value clipping tracked values
        self.max = -1e8
        self.min = 1e8
        self.max_target = 0.0
        self.min_target = 0.0

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

        # Encoder optimiser
        self.encoder_net_optimiser = torch.optim.Adam(
            self.encoder_net.parameters(),
            lr=config.encoder_lr,
            **config.encoder_lr_params,
        )

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        self.actor_net.eval()

        state = action_context.state
        evaluation = action_context.evaluation

        assert isinstance(state, np.ndarray)

        with torch.no_grad():
            # Fix: Use modern tensor creation
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
            state_tensor = state_tensor.unsqueeze(0)

            if evaluation:
                zs = self.checkpoint_encoder.zs(state_tensor)
                action = self.checkpoint_actor(state_tensor, zs)
            else:
                zs = self.fixed_encoder_net.zs(state_tensor)
                action = self.actor_net(state_tensor, zs)

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
        # Fix: Use modern tensor creation
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device)
        state_tensor = state_tensor.unsqueeze(0)

        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device)
        action_tensor = action_tensor.unsqueeze(0)

        with torch.no_grad():
            with hlp.evaluating(self.critic_net):
                # Fix: Use proper TD7 critic interface with encodings
                fixed_zs = self.fixed_encoder_net.zs(state_tensor)
                fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, action_tensor)

                q_values_one, q_values_two = self.critic_net(
                    state_tensor, action_tensor, fixed_zsa, fixed_zs
                )
                q_value = torch.minimum(q_values_one, q_values_two)

        return q_value[0].item()

    def _update_encoder(
        self, states: torch.Tensor, actions: torch.Tensor, next_states: torch.Tensor
    ) -> dict[str, Any]:

        with torch.no_grad():
            next_zs = self.encoder_net.zs(next_states)

        zs = self.encoder_net.zs(states)
        pred_zs = self.encoder_net.zsa(zs, actions)

        encoder_loss = F.mse_loss(pred_zs, next_zs)

        self.encoder_net_optimiser.zero_grad()
        encoder_loss.backward()
        self.encoder_net_optimiser.step()

        info = {"encoder_loss": encoder_loss.item()}
        return info

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
            fixed_target_zs = self.target_fixed_encoder_net.zs(next_states)

            next_actions = self.target_actor_net(next_states, fixed_target_zs)

            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(
                target_noise, -self.policy_noise_clip, self.policy_noise_clip
            )

            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            fixed_target_zsa = self.target_fixed_encoder_net.zsa(
                fixed_target_zs, next_actions
            )

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions, fixed_target_zsa, fixed_target_zs
            )

            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            target_q_values = target_q_values.clamp(self.min_target, self.max_target)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

            self.max = max(self.max, float(q_target.max()))
            self.min = min(self.min, float(q_target.min()))

            fixed_zs = self.fixed_encoder_net.zs(states)
            fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, actions)

        q_values_one, q_values_two = self.critic_net(
            states, actions, fixed_zsa, fixed_zs
        )

        td_error_one = (q_values_one - q_target).abs()
        td_error_two = (q_values_two - q_target).abs()

        huber_loss_one = hlp.calculate_huber_loss(
            td_error_one,
            self.min_priority,
            use_quadratic_smoothing=False,
            use_mean_reduction=False,
        )
        huber_loss_two = hlp.calculate_huber_loss(
            td_error_two,
            self.min_priority,
            use_quadratic_smoothing=False,
            use_mean_reduction=False,
        )

        critic_loss_total = (huber_loss_one + huber_loss_two).mean()

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
            "critic_loss_one": huber_loss_one.mean().item(),
            "critic_loss_two": huber_loss_two.mean().item(),
            "critic_loss_total": critic_loss_total.item(),
        }

        return info, priorities

    # Weights is set for methods like MAPERTD3 that use weights in the actor update
    def _update_actor(
        self,
        states: torch.Tensor,
        weights: torch.Tensor,  # pylint: disable=unused-argument
    ) -> dict[str, Any]:
        with hlp.evaluating(self.encoder_net):
            fixed_zs = self.fixed_encoder_net.zs(states)

        actions = self.actor_net(states, fixed_zs)
        fixed_zsa = self.fixed_encoder_net.zsa(fixed_zs, actions)

        with hlp.evaluating(self.critic_net):
            actor_q_values_one, actor_q_values_two = self.critic_net(
                states, actions, fixed_zsa, fixed_zs
            )

        # Concatenate both Q-values then take mean (like reference TD7)
        actor_q_values = torch.cat([actor_q_values_one, actor_q_values_two], dim=1)
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

        encoder_info = self._update_encoder(
            states_tensor, actions_tensor, next_states_tensor
        )
        info |= encoder_info

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

        # Update the Priorities
        if self.use_per_buffer:
            memory.update_priorities(indices, priorities)

        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            # Update the Actor and Alpha
            actor_info = self._update_actor(states_tensor, weights_tensor)
            info |= actor_info

        if self.learn_counter % self.target_update_freq == 0:
            # Update target network params
            hlp.soft_update_params(self.critic_net, self.target_critic_net, self.tau)
            hlp.soft_update_params(self.actor_net, self.target_actor_net, self.tau)

            hlp.soft_update_params(
                self.fixed_encoder_net, self.target_fixed_encoder_net, self.tau
            )
            hlp.soft_update_params(self.encoder_net, self.fixed_encoder_net, self.tau)

            memory.reset_max_priority()

            self.max_target = self.max
            self.min_target = self.min

        return info

    # TODO use training_step with decay rates
    def _train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
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

    def _train_and_reset(self, training_context: TrainingContext) -> dict[str, Any]:
        info: dict[str, Any] = {}

        for _ in range(self.timesteps_since_update):
            if self.learn_counter == self.steps_before_checkpointing:
                self.best_min_return *= self.reset_weight
                self.max_eps_before_update = self.max_eps_checkpointing

            info = self._train_policy(training_context)

        self.eps_since_update = 0
        self.timesteps_since_update = 0
        self.min_return = 1e8

        return info

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        info: dict[str, Any] = {}

        episode_steps = training_context.episode_steps
        episode_return = training_context.episode_reward
        episode_done = training_context.episode_done

        if not episode_done:
            return info

        self.eps_since_update += 1
        self.timesteps_since_update += episode_steps

        self.min_return = min(self.min_return, episode_return)

        if self.min_return < self.best_min_return:
            info = self._train_and_reset(training_context)

        elif self.eps_since_update == self.max_eps_before_update:
            self.best_min_return = self.min_return
            self.checkpoint_actor.load_state_dict(self.actor_net.state_dict())
            self.checkpoint_encoder.load_state_dict(self.encoder_net.state_dict())

            info = self._train_and_reset(training_context)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "encoder": self.encoder_net.state_dict(),
            "target_actor": self.target_actor_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "target_fixed_encoder": self.target_fixed_encoder_net.state_dict(),
            "fixed_encoder": self.fixed_encoder_net.state_dict(),
            "checkpoint_actor": self.checkpoint_actor.state_dict(),
            "checkpoint_encoder": self.checkpoint_encoder.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "encoder_optimizer": self.encoder_net_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
            "policy_noise": self.policy_noise,
            "action_noise": self.action_noise,
            # Add value tracking
            "max": self.max,
            "min": self.min,
            "max_target": self.max_target,
            "min_target": self.min_target,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.target_actor_net.load_state_dict(checkpoint["target_actor"])

        self.critic_net.load_state_dict(checkpoint["critic"])
        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        # Load encoder networks
        self.encoder_net.load_state_dict(checkpoint["encoder"])
        self.target_fixed_encoder_net.load_state_dict(
            checkpoint["target_fixed_encoder"]
        )
        self.fixed_encoder_net.load_state_dict(checkpoint["fixed_encoder"])
        self.checkpoint_actor.load_state_dict(checkpoint["checkpoint_actor"])
        self.checkpoint_encoder.load_state_dict(checkpoint["checkpoint_encoder"])

        # Load optimizers
        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        self.encoder_net_optimiser.load_state_dict(checkpoint["encoder_optimizer"])

        # Load training state
        self.learn_counter = checkpoint.get("learn_counter", 0)
        self.policy_noise = checkpoint.get("policy_noise", self.policy_noise)
        self.action_noise = checkpoint.get("action_noise", self.action_noise)

        # Load value tracking
        self.max = checkpoint.get("max", -1e8)
        self.min = checkpoint.get("min", 1e8)
        self.max_target = checkpoint.get("max_target", 0)
        self.min_target = checkpoint.get("min_target", 0)

        logging.info("models, optimisers, and training state have been loaded...")
