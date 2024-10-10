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
from cares_reinforcement_learning.encoders.configurations import VanillaAEConfig
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.memory import MemoryBuffer


class SACDAE:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        decoder_network: torch.nn.Module,
        gamma: float,
        tau: float,
        reward_scale: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        alpha_lr: float,
        target_entropy_multiplier: float,
        encoder_tau: float,
        decoder_update_freq: int,
        ae_config: VanillaAEConfig,
        device: torch.device,
    ):
        self.type = "discrete_policy"
        self.device = device

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        # tie the encoder weights
        self.actor_net.encoder.copy_conv_weights_from(self.critic_net.encoder)

        self.encoder_tau = encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = ae_config.latent_lambda

        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        self.learn_counter = 0
        self.policy_update_freq = 2
        self.target_update_freq = 2

        actor_beta = 0.9
        critic_beta = 0.9
        alpha_beta = 0.5

        target_entropy_multiplier = 0.7

        self.target_entropy = np.log(1.0 / action_num) * target_entropy_multiplier

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.loss_function = AELoss(latent_lambda=ae_config.latent_lambda)

        self.encoder_net_optimiser = torch.optim.Adam(
            self.critic_net.encoder.parameters(), **ae_config.encoder_optim_kwargs
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            **ae_config.decoder_optim_kwargs,
        )
        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 0.1 according to other baselines.
        init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0, info: dict[str, any] = {}
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            state_tensor = state_tensor / 255


            sample_action, action_probs, _, max_prob_action = self.actor_net(state_tensor)

            info["sample_action"] = sample_action.item()
            info["action_probabilities"] = action_probs[0].cpu().numpy()
            info["max_prob_action"] = max_prob_action.item()

            action = max_prob_action if evaluation else sample_action

        self.actor_net.train()
        return action.item()

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
    ) -> tuple[float, float, float]:
        with torch.no_grad():
            _, action_probs, log_actions_probs, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states
            )

            target_q_values = action_probs * (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * log_actions_probs
            )

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        return critic_loss_one.item(), critic_loss_two.item(), critic_loss_total.item()

    def _update_actor_alpha(self, states: torch.Tensor) -> tuple[float, float]:
        _, action_probs, log_action_probs, _ = self.actor_net(
            states, detach_encoder=True
        )

        qf1_pi, qf2_pi = self.critic_net(states, detach_encoder=True)
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

        return actor_loss.item(), alpha_loss.item()

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

        return ae_loss.item()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.LongTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # Normalise states and next_states
        # This because the states are [0-255] and the predictions are [0-1]
        states_normalised = states / 255
        next_states_normalised = next_states / 255

        info = {}

        # Update the Critic
        critic_loss_one, critic_loss_two, critic_loss_total = self._update_critic(
            states_normalised, actions, rewards, next_states_normalised, dones
        )
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["critic_loss"] = critic_loss_total

        # Update the Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_loss, alpha_loss = self._update_actor_alpha(states_normalised)
            info["actor_loss"] = actor_loss
            info["alpha_loss"] = alpha_loss
            info["alpha"] = self.alpha.item()

        if self.learn_counter % self.target_update_freq == 0:
            # Update the target networks - Soft Update
            hlp.soft_update_params(
                self.critic_net.Q1, self.target_critic_net.Q1, self.tau
            )
            hlp.soft_update_params(
                self.critic_net.Q2, self.target_critic_net.Q2, self.tau
            )
            hlp.soft_update_params(
                self.critic_net.encoder,
                self.target_critic_net.encoder,
                self.encoder_tau,
            )

        if self.learn_counter % self.decoder_update_freq == 0:
            ae_loss = self._update_autoencoder(states_normalised)
            info["ae_loss"] = ae_loss

        return info

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        torch.save(self.decoder_net.state_dict(), f"{path}/{filename}_decoder.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        self.decoder_net.load_state_dict(torch.load(f"{path}/{filename}_decoder.pht"))
        logging.info("models has been loaded...")
