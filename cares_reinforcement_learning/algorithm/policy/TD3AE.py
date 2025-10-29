"""
Original Paper: https://arxiv.org/abs/1910.01741 - SAC based but followed same concept here
Code based on: https://github.com/denisyarats/pytorch_sac_ae/tree/master
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
from cares_reinforcement_learning.algorithm.algorithm import ImageAlgorithm
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
from cares_reinforcement_learning.networks.TD3AE import Actor, Critic
from cares_reinforcement_learning.util.configurations import TD3AEConfig
from cares_reinforcement_learning.util.training_context import (
    ActionContext,
    TrainingContext,
)


class TD3AE(ImageAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        decoder_network: Decoder,
        config: TD3AEConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net).to(self.device)
        self.target_actor_net.eval()  # never in training mode - helps with batch/drop out layers
        self.target_critic_net = copy.deepcopy(self.critic_net).to(self.device)
        self.target_critic_net.eval()  # never in training mode - helps with batch/drop out layers

        # tie the encoder weights
        self.actor_net.encoder.copy_conv_weights_from(self.critic_net.encoder)

        self.encoder_tau = config.encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_latent_lambda = config.autoencoder_config.latent_lambda
        self.decoder_update_freq = config.decoder_update_freq

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

        self.ae_loss_function = AELoss(
            latent_lambda=config.autoencoder_config.latent_lambda
        )

        self.encoder_net_optimiser = torch.optim.Adam(
            self.critic_net.encoder.parameters(),
            **config.autoencoder_config.encoder_optim_kwargs,
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            **config.autoencoder_config.decoder_optim_kwargs,
        )

    def select_action_from_policy(self, action_context: ActionContext) -> np.ndarray:
        self.actor_net.eval()

        state = action_context.state
        evaluation = action_context.evaluation

        assert isinstance(state, dict)

        with torch.no_grad():
            state_tensor = tu.image_state_to_tensors(state, self.device)

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

    def _update_critic(
        self,
        states: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: dict[str, torch.Tensor],
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

        td_error_one = (q_values_one.detach() - q_target).abs()
        td_error_two = (q_values_two.detach() - q_target).abs()

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

    def _update_actor(self, states: dict[str, torch.Tensor]) -> dict[str, Any]:
        actions = self.actor_net(states, detach_encoder=True)

        with hlp.evaluating(self.critic_net):
            actor_q_values, _ = self.critic_net(states, actions, detach_encoder=True)

        actor_loss = -actor_q_values.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        info = {
            "actor_loss": actor_loss.item(),
        }
        return info

    def _update_autoencoder(self, states: torch.Tensor) -> dict[str, Any]:
        latent_samples = self.critic_net.encoder(states)
        reconstructed_data = self.decoder_net(latent_samples)

        ae_loss = self.ae_loss_function.calculate_loss(
            data=states,
            reconstructed_data=reconstructed_data,
            latent_sample=latent_samples,
        )

        self.encoder_net_optimiser.zero_grad()
        self.decoder_net_optimiser.zero_grad()
        ae_loss.backward()
        self.encoder_net_optimiser.step()
        self.decoder_net_optimiser.step()

        info = {
            "ae_loss": ae_loss.item(),
        }
        return info

    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]:
        self.learn_counter += 1

        memory = training_context.memory
        batch_size = training_context.batch_size

        self.policy_noise *= self.policy_noise_decay
        self.policy_noise = max(self.min_policy_noise, self.policy_noise)

        self.action_noise *= self.action_noise_decay
        self.action_noise = max(self.min_action_noise, self.action_noise)

        # Sample and convert to tensors using multimodal sampling
        (
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
            weights_tensor,
            indices,
        ) = tu.sample_image_batch_to_tensors(
            memory,
            batch_size,
            self.device,
            use_per_buffer=self.use_per_buffer,
            per_sampling_strategy=self.per_sampling_strategy,
            per_weight_normalisation=self.per_weight_normalisation,
        )

        info: dict[str, Any] = {}

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
            actor_info = self._update_actor(states_tensor)
            info |= actor_info

            # Update target network params
            hlp.soft_update_params(
                self.critic_net.critic.Q1,
                self.target_critic_net.critic.Q1,
                self.tau,
            )
            hlp.soft_update_params(
                self.critic_net.critic.Q2,
                self.target_critic_net.critic.Q2,
                self.tau,
            )

            hlp.soft_update_params(
                self.critic_net.encoder,
                self.target_critic_net.encoder,
                self.encoder_tau,
            )

            hlp.soft_update_params(
                self.actor_net.actor.act_net,
                self.target_actor_net.actor.act_net,
                self.encoder_tau,
            )

            hlp.soft_update_params(
                self.actor_net.encoder, self.target_actor_net.encoder, self.encoder_tau
            )

        if self.learn_counter % self.decoder_update_freq == 0:
            ae_info = self._update_autoencoder(states_tensor["image"])
            info |= ae_info

        # Update the Priorities
        if self.use_per_buffer:
            memory.update_priorities(indices, priorities)

        return info

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)
        checkpoint = {
            "actor": self.actor_net.state_dict(),
            "critic": self.critic_net.state_dict(),
            "target_actor": self.target_actor_net.state_dict(),
            "target_critic": self.target_critic_net.state_dict(),
            "decoder": self.decoder_net.state_dict(),
            "actor_optimizer": self.actor_net_optimiser.state_dict(),
            "critic_optimizer": self.critic_net_optimiser.state_dict(),
            "encoder_optimizer": self.encoder_net_optimiser.state_dict(),
            "decoder_optimizer": self.decoder_net_optimiser.state_dict(),
            "learn_counter": self.learn_counter,
            "policy_noise": self.policy_noise,
            "action_noise": self.action_noise,
        }
        torch.save(checkpoint, f"{filepath}/{filename}_checkpoint.pth")
        logging.info("models, optimisers, and training state have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        checkpoint = torch.load(f"{filepath}/{filename}_checkpoint.pth")

        self.actor_net.load_state_dict(checkpoint["actor"])
        self.critic_net.load_state_dict(checkpoint["critic"])

        self.target_actor_net.load_state_dict(checkpoint["target_actor"])
        self.target_critic_net.load_state_dict(checkpoint["target_critic"])

        self.decoder_net.load_state_dict(checkpoint["decoder"])

        self.actor_net_optimiser.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_net_optimiser.load_state_dict(checkpoint["critic_optimizer"])
        self.encoder_net_optimiser.load_state_dict(checkpoint["encoder_optimizer"])
        self.decoder_net_optimiser.load_state_dict(checkpoint["decoder_optimizer"])

        self.learn_counter = checkpoint.get("learn_counter", 0)

        self.policy_noise = checkpoint.get("policy_noise", self.policy_noise)
        self.action_noise = checkpoint.get("action_noise", self.action_noise)
        logging.info("models, optimisers, and training state have been loaded...")
