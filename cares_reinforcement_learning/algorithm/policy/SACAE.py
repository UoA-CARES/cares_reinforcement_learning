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

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer


class SACAE:
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
        encoder_lr: float,
        encoder_tau: float,
        decoder_lr: float,
        decoder_latent_lambda: float,
        decoder_weight_decay: float,
        decoder_update_freq: int,
        alpha_lr: float,
        device: torch.device,
        is_1d: bool = False
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

        self.encoder_tau = encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_latent_lambda = decoder_latent_lambda
        self.decoder_update_freq = decoder_update_freq

        self.gamma = gamma
        self.tau = tau
        self.reward_scale = reward_scale

        self.learn_counter = 0
        self.policy_update_freq = 2
        self.target_update_freq = 2

        actor_beta = 0.9
        critic_beta = 0.9
        alpha_beta = 0.5

        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_num)

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.encoder_net_optimiser = torch.optim.Adam(
            self.critic_net.encoder.parameters(), lr=encoder_lr
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_decay,
        )

        # needed since tensor shapes need to be treated differently
        self.is_1d = is_1d

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
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0
    ) -> np.ndarray:
        # note that when evaluating this algorithm we need to select mu as action
        self.actor_net.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state)

            # normally input of shape [3,w,h] to [1,3,w,h] to account for batch size
            # somehow in 1d case channel size of 1 is ommited, need to unsqueeze twice to get [1,1,num_of_features]
            if self.is_1d:
                state_tensor = state_tensor.unsqueeze(0).unsqueeze(0).to(self.device)
            else:
                state_tensor = state_tensor.unsqueeze(0).to(self.device)
            
            #TODO: Doesn't make sense for non-image input, but not breaking
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

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
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

    def _update_actor_alpha(self, states: torch.Tensor) -> None:
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

    def _update_autoencoder(self, states: torch.Tensor) -> None:
        states_latent = self.critic_net.encoder(states)
        rec_observations = self.decoder_net(states_latent)

        rec_loss = F.mse_loss(states, rec_observations)

        # add L2 penalty on latent representation
        # see https://arxiv.org/pdf/1903.12436.pdf
        latent_loss = (0.5 * states_latent.pow(2).sum(1)).mean()

        loss = rec_loss + self.decoder_latent_lambda * latent_loss
        self.encoder_net_optimiser.zero_grad()
        self.decoder_net_optimiser.zero_grad()
        loss.backward()
        self.encoder_net_optimiser.step()
        self.decoder_net_optimiser.step()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
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

        # Here since passing states directly in result in shape [1,batch_size,num_of_features] SOMEHOW
        # might be related to that weird omitting size of 1 issue
        if self.is_1d:
            states = states.view(batch_size,1,-1)
            next_states = next_states.view(batch_size,1,-1)

        # Reshape to batch_size x whatever
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        #TODO: does not make sense for non-image cases. However some scaling does not break anything either.
        # Normalise states and next_states
        # This because the states are [0-255] and the predictions are [0-1]
        states_normalised = states / 255
        next_states_normalised = next_states / 255

        # Update the Critic
        self._update_critic(
            states_normalised, actions, rewards, next_states_normalised, dones
        )

        # Update the Actor
        if self.learn_counter % self.policy_update_freq == 0:
            self._update_actor_alpha(states_normalised)

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
            self._update_autoencoder(states_normalised)

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
