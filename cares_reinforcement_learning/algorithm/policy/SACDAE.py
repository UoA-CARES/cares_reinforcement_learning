"""
Original SACD Paper: https://arxiv.org/pdf/1910.07207
Original SACAE Paper: https://arxiv.org/abs/1910.01741
Code based on UoA-CARES SACD and SACAE implementations.

This code runs automatic entropy tuning
"""

import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.algorithm.algorithm import ImageAlgorithm
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.SACDAE import Actor, Critic
from cares_reinforcement_learning.util.configurations import SACDAEConfig



class SACDAE(ImageAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        decoder_network: Decoder,
        config: SACDAEConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="discrete_policy", config=config, device=device)

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)
        self.target_critic_net.eval()

        # tie the encoder weights
        self.actor_net.encoder.copy_conv_weights_from(self.critic_net.encoder)

        self.encoder_tau = config.encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_update_freq = config.decoder_update_freq
        self.decoder_latent_lambda = config.autoencoder_config.latent_lambda

        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        # PER
        self.use_per_buffer = config.use_per_buffer
        self.per_sampling_strategy = config.per_sampling_strategy
        self.per_weight_normalisation = config.per_weight_normalisation
        self.per_alpha = config.per_alpha
        self.min_priority = config.min_priority

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq
        self.target_update_freq = config.target_update_freq

        # self.target_entropy = -np.prod(
        #     np.log(self.actor_net.num_actions) * config.target_entropy_multiplier
        # ) - 1
        self.target_entropy = -self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, **config.actor_lr_params
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr, **config.critic_lr_params
        )

        self.loss_function = AELoss(latent_lambda=config.autoencoder_config.latent_lambda)

        self.encoder_net_optimiser = torch.optim.Adam(
            self.critic_net.encoder.parameters(), **config.autoencoder_config.encoder_optim_kwargs
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            **config.autoencoder_config.decoder_optim_kwargs,
        )
        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 0.1 according to other baselines.
        init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr, **config.alpha_lr_params
        )

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = hlp.image_state_dict_to_tensor(state, self.device)

            if evaluation:
                (_, _, action) = self.actor_net(state_tensor)
            else:
                (action, _, _) = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
        self.actor_net.train()
        return action

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

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
                _, (action_probs, log_actions_probs), next_actions = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, []
            )

            temp_min_qf_next_target = action_probs * (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * log_actions_probs
            )
            target_q_values = temp_min_qf_next_target

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, [])

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

    def _update_actor_alpha(self, states: torch.Tensor) -> tuple[float, float]:
        _, (action_probs, log_action_probs), _ = self.actor_net(
            states, detach_encoder=True
        )

        qf1_pi, qf2_pi = self.critic_net(states, [], detach_encoder=True)
        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)

        inside_term = self.alpha * log_action_probs - min_qf_pi
        actor_loss = (action_probs * inside_term).sum(dim=1).mean()

        log_action_probs = torch.sum(log_action_probs * action_probs, dim=1)

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # Update the temperature (alpha)
        alpha_loss = -(
            self.log_alpha * (log_action_probs + self.target_entropy).detach()
        ).mean()

        self.log_alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        info = {
            "actor_loss": actor_loss.item(),
            "alpha_loss": alpha_loss.item(),
        }

        return info

    def _update_autoencoder(self, states: torch.Tensor) -> float:
        latent_samples = self.critic_net.encoder(states)
        reconstructed_data = self.decoder_net(latent_samples)

        ae_loss = self.loss_function.calculate_loss(
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

    def train_policy(self, memory: MemoryBuffer, batch_size: int, training_step: int) -> dict[str, Any]:
        self.learn_counter += 1

        if self.use_per_buffer:
            experiences = memory.sample_priority(
                batch_size,
                sampling_strategy=self.per_sampling_strategy,
                weight_normalisation=self.per_weight_normalisation,
            )
            states, actions, rewards, next_states, dones, indices, priorities = experiences
        else:
            experiences = memory.sample_uniform(batch_size)
            states, actions, rewards, next_states, dones, _ = experiences
            weights = [1.0] * batch_size

        batch_size = len(states)

        states_tensor = hlp.image_states_dict_to_tensor(states, self.device)

        actions_tensor = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)

        next_states_tensor = hlp.image_states_dict_to_tensor(next_states, self.device)

        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)
        weights_tensor = torch.FloatTensor(np.asarray(weights)).to(self.device)

        # Reshape to batch_size x whatever
        # actions_tensor = actions_tensor.reshape(batch_size, 1)
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

        # Update the Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_info = self._update_actor_alpha(states_tensor)
            info |= actor_info
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            # Update the target networks - Soft Update
            hlp.soft_update_params(
                self.critic_net.critic.Q1, self.target_critic_net.critic.Q1, self.tau
            )
            hlp.soft_update_params(
                self.critic_net.critic.Q2, self.target_critic_net.critic.Q2, self.tau
            )
            hlp.soft_update_params(
                self.critic_net.encoder,
                self.target_critic_net.encoder,
                self.encoder_tau,
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
        torch.save(self.actor_net.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{filepath}/{filename}_critic.pht")
        torch.save(self.decoder_net.state_dict(), f"{filepath}/{filename}_decoder.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor_net.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht"))
        self.decoder_net.load_state_dict(
            torch.load(f"{filepath}/{filename}_decoder.pht")
        )
        logging.info("models has been loaded...")
