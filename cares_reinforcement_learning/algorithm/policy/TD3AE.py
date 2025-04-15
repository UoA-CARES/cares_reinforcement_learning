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
from cares_reinforcement_learning.algorithm.algorithm import ImageAlgorithm
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.encoders.vanilla_autoencoder import Decoder
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.TD3AE import Actor, Critic
from cares_reinforcement_learning.util.configurations import TD3AEConfig


class TD3AE(ImageAlgorithm):
    def __init__(
        self,
        actor_network: Actor,
        critic_network: Critic,
        decoder_network: Decoder,
        config: TD3AEConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", device=device)

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

        self.noise_clip = config.policy_noise_clip
        self.policy_noise = config.policy_noise

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq

        self.action_num = self.actor_net.num_actions

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=config.actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=config.critic_lr
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

    def select_action_from_policy(
        self,
        state: dict[str, np.ndarray],
        evaluation: bool = False,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = hlp.image_state_dict_to_tensor(state, self.device)

            action = self.actor_net(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
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
    ) -> tuple[float, float, float]:
        with torch.no_grad():
            next_actions = self.target_actor_net(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states, next_actions
            )

            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic_net(states, actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        # Update the Critic
        self.critic_net_optimiser.zero_grad()
        critic_loss_total.backward()
        self.critic_net_optimiser.step()

        return critic_loss_one.item(), critic_loss_two.item(), critic_loss_total.item()

    def _update_actor(self, states: dict[str, torch.Tensor]) -> float:
        actions = self.actor_net(states, detach_encoder=True)

        with hlp.evaluating(self.critic_net):
            actor_q_values, _ = self.critic_net(states, actions, detach_encoder=True)

        actor_loss = -actor_q_values.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        return actor_loss.item()

    def _update_autoencoder(self, states: torch.Tensor) -> float:
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

        return ae_loss.item()

    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        states_tensor = hlp.image_states_dict_to_tensor(states, self.device)

        actions_tensor = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards_tensor = torch.FloatTensor(np.asarray(rewards)).to(self.device)

        next_states_tensor = hlp.image_states_dict_to_tensor(next_states, self.device)

        dones_tensor = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards_tensor = rewards_tensor.reshape(batch_size, 1)
        dones_tensor = dones_tensor.reshape(batch_size, 1)

        info = {}

        critic_loss_one, critic_loss_two, critic_loss_total = self._update_critic(
            states_tensor,
            actions_tensor,
            rewards_tensor,
            next_states_tensor,
            dones_tensor,
        )
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["critic_loss"] = critic_loss_total

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_loss = self._update_actor(states_tensor)
            info["actor_loss"] = actor_loss

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
            ae_loss = self._update_autoencoder(states_tensor["image"])
            info["ae_loss"] = ae_loss

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
