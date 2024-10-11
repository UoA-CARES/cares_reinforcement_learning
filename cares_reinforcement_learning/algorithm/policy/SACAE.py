"""
Original Paper: https://arxiv.org/abs/1910.01741
Code based on: https://github.com/denisyarats/pytorch_sac_ae/tree/master

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
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.configurations import SACAEConfig


class SACAE:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        decoder_network: Decoder,
        config: SACAEConfig,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        # tie the encoder weights
        self.actor_net.encoder.copy_conv_weights_from(self.critic_net.encoder)

        self.encoder_tau = config.encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_update_freq = config.decoder_update_freq
        self.decoder_latent_lambda = config.autoencoder_config.latent_lambda

        self.gamma = config.gamma
        self.tau = config.tau
        self.reward_scale = config.reward_scale

        self.learn_counter = 0
        self.policy_update_freq = 2
        self.target_update_freq = 2

        actor_beta = 0.9
        critic_beta = 0.9
        alpha_beta = 0.5

        # set target entropy to -|A|
        self.target_entropy = -np.prod(self.actor_net.num_actions)

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(),
            lr=config.critic_lr,
            betas=(critic_beta, 0.999),
        )

        self.loss_function = AELoss(
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

        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 0.1 according to other baselines.
        init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=config.alpha_lr, betas=(alpha_beta, 0.999)
        )

    def select_action_from_policy(
        self,
        state: dict[str, np.ndarray],
        evaluation: bool = False,
        noise_scale: float = 0,
    ) -> np.ndarray:
        # pylint: disable-next=unused-argument

        # note that when evaluating this algorithm we need to select mu as action
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
        states: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[float, float, float]:

        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )
            target_q_values = (
                torch.minimum(target_q_values_one, target_q_values_two)
                - self.alpha * next_log_pi
            )

            q_target = (
                rewards * self.reward_scale + self.gamma * (1 - dones) * target_q_values
            )

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        return critic_loss_one.item(), critic_loss_two.item(), critic_loss_total.item()

    def _update_actor_alpha(
        self, states: dict[str, torch.Tensor]
    ) -> tuple[float, float]:
        pi, log_pi, _ = self.actor_net(states, detach_encoder=True)
        qf1_pi, qf2_pi = self.critic_net(states, pi, detach_encoder=True)

        min_qf_pi = torch.minimum(qf1_pi, qf2_pi)
        actor_loss = ((self.alpha * log_pi) - min_qf_pi).mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        # Update the temperature (alpha)
        alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()

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

        states = hlp.image_states_dict_to_tensor(states, self.device)

        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)

        next_states = hlp.image_states_dict_to_tensor(next_states, self.device)

        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        info = {}

        # Update the Critic
        critic_loss_one, critic_loss_two, critic_loss_total = self._update_critic(
            states, actions, rewards, next_states, dones
        )
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["critic_loss"] = critic_loss_total

        # Update the Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_loss, alpha_loss = self._update_actor_alpha(states)
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
            ae_loss = self._update_autoencoder(states["image"])
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
