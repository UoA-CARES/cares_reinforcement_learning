"""
Original Paper: https://arxiv.org/abs/1910.01741
Code based on: https://github.com/denisyarats/pytorch_sac_ae/tree/master

This code runs automatic entropy tuning
"""

import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
import cares_reinforcement_learning.util.helpers as hlp


class SACAE:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        encoder_network: torch.nn.Module,
        decoder_network: torch.nn.Module,
        gamma: float,
        tau: float,
        reward_scale: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        encoder_lr: float,
        encoder_tau: float,
        decoder_lr: float,
        decoder_latent_lambda: float,
        decoder_weight_decay: float,
        decoder_update_freq: int,
        alpha_lr: float,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        # this may be called policy_net in other implementations
        self.actor_net = actor_network.to(device)

        # this may be called soft_q_net in other implementations
        self.critic_net = critic_network.to(device)
        self.target_critic_net = copy.deepcopy(self.critic_net).to(device)

        self.encoder_net_orig = copy.deepcopy(encoder_network).to(device)

        self.encoder_net = encoder_network.to(device)
        self.encoder_tau = encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder_update_freq = decoder_update_freq

        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        self.learn_counter = 0
        self.policy_update_freq = 1

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_num)

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        self.encoder_net_optimiser = torch.optim.Adam(
            self.encoder_net.parameters(), lr=encoder_lr
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_decay,
        )

        # Temperature (alpha) for the entropy loss
        # Set to initial alpha to 0.1 according to other baselines.
        init_temperature = 0.1
        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=alpha_lr)

    # pylint: disable-next=unused-argument
    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)
            state_tensor = state_tensor.unsqueeze(0).to(self.device)
            state_tensor = state_tensor / 255

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

    def train_policy(self, memory: PrioritizedReplayBuffer, batch_size: int) -> None:
        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        # Convert into tensor
        states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # Normalise states and next_states
        states_normalised = states / 255
        next_states_normalised = next_states / 255

        # Update the Critic
        with torch.no_grad():
            next_actions, next_log_pi, _ = self.actor_net(next_states_normalised)

            target_q_values_one, target_q_values_two = self.target_critic_net(
                next_states_normalised, next_actions
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

        # Update the Actor
        pi, log_pi, _ = self.actor_net(states_normalised, detach_encoder=True)
        qf1_pi, qf2_pi = self.critic_net(states_normalised, pi, detach_encoder=True)

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

        if self.learn_counter % self.policy_update_freq == 0:

            # Update the target networks - Soft Update
            # TODO only update Q1 and Q2 not the encoder/decoder
            for target_param, param in zip(
                self.target_critic_net.parameters(), self.critic_net.parameters()
            ):
                target_param.data.copy_(
                    param.data * self.tau + target_param.data * (1.0 - self.tau)
                )

        if self.learn_counter % self.decoder_update_freq == 0:
            states_latent = self.encoder_net(states_normalised)
            rec_observations = self.decoder_net(states_latent)

            rec_loss = F.mse_loss(states_normalised, rec_observations)

            # add L2 penalty on latent representation
            # see https://arxiv.org/pdf/1903.12436.pdf
            latent_loss = (0.5 * states_latent.pow(2).sum(1)).mean()

            loss = rec_loss + self.decoder_latent_lambda * latent_loss
            self.encoder_net_optimiser.zero_grad()
            self.decoder_net_optimiser.zero_grad()
            loss.backward()
            self.encoder_net_optimiser.step()
            self.decoder_net_optimiser.step()

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)

        torch.save(self.actor_net.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic_net.state_dict(), f"{path}/{filename}_critic.pht")
        torch.save(self.encoder_net.state_dict(), f"{path}/{filename}_encoder.pht")
        torch.save(self.decoder_net.state_dict(), f"{path}/{filename}_decoder.pht")
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        self.encoder_net.load_state_dict(torch.load(f"{path}/{filename}_encoder.pht"))
        self.decoder_net.load_state_dict(torch.load(f"{path}/{filename}_decoder.pht"))
        logging.info("models has been loaded...")