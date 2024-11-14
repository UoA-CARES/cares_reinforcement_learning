import copy
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

# This is used to metric the novelty.
from skimage.metrics import structural_similarity as ssim
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.memory import MemoryBuffer

# TODO no sure how to import this, the ensemble will be the same? Can I pass this form outside?
from cares_reinforcement_learning.networks.NaSATD3.EPDM import EPDM


class NaSATD3:
    def __init__(
        self,
        encoder_network: nn.Module,
        decoder_network: nn.Module,
        actor_network: nn.Module,
        critic_network: nn.Module,
        gamma: float,
        tau: float,
        ensemble_size: int,
        action_num: int,
        latent_size: int,
        intrinsic_on: bool,
        actor_lr: float,
        critic_lr: float,
        encoder_lr: float,
        decoder_lr: float,
        epm_lr: float,
        device: str,
    ):
        self.type = "policy"
        self.device = device

        self.gamma = gamma
        self.tau = tau

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.ensemble_size = ensemble_size
        self.latent_size = latent_size
        self.intrinsic_on = intrinsic_on

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num

        self.encoder = encoder_network.to(device)
        self.decoder = decoder_network.to(device)
        self.actor = actor_network.to(device)
        self.critic = critic_network.to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        # Necessary for make the same encoder in the whole algorithm
        self.actor_target.encoder_net = self.encoder
        self.critic_target.encoder_net = self.encoder

        self.ensemble_predictive_model = nn.ModuleList()
        networks = [
            EPDM(self.latent_size, self.action_num) for _ in range(self.ensemble_size)
        ]
        self.ensemble_predictive_model.extend(networks)
        self.ensemble_predictive_model.to(self.device)

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )

        self.encoder_lr = encoder_lr
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(), lr=self.encoder_lr
        )

        self.decoder_lr = decoder_lr
        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(), lr=self.decoder_lr, weight_decay=1e-7
        )

        self.epm_lr = epm_lr
        self.epm_optimizers = [
            torch.optim.Adam(
                self.ensemble_predictive_model[i].parameters(),
                lr=self.epm_lr,
                weight_decay=1e-3,
            )
            for i in range(self.ensemble_size)
        ]

    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False, noise_scale: float = 0.1
    ) -> np.ndarray:
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor / 255

            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                action += noise_scale * np.random.randn(self.action_num)
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)

        self.actor.train()
        return action

    def _update_critic(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_noise = self.policy_noise * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -self.noise_clip, self.noise_clip)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.critic_target(
                next_states, next_actions
            )
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic(states, actions)

        critic_loss_one = F.mse_loss(q_values_one, q_target)
        critic_loss_two = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_one + critic_loss_two

        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

    def _update_autoencoder(self, states: torch.Tensor) -> None:
        z_vector = self.encoder(states)
        rec_obs = self.decoder(z_vector)

        target_images = states
        rec_loss = F.mse_loss(target_images, rec_obs)

        # add L2 penalty on latent representation
        latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()
        ae_loss = rec_loss + 1e-6 * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def _update_actor(self, states: torch.Tensor) -> None:
        actor_q_one, actor_q_two = self.critic(
            states, self.actor(states, detach_encoder=True), detach_encoder=True
        )
        actor_q_values = torch.minimum(actor_q_one, actor_q_two)
        actor_loss = -actor_q_values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def _update_predictive_model(
        self, states: np.ndarray, actions: np.ndarray, next_states: np.ndarray
    ) -> None:

        with torch.no_grad():
            latent_state = self.encoder(states, detach_output=True)
            latent_next_state = self.encoder(next_states, detach_output=True)

        for predictive_network, optimizer in zip(
            self.ensemble_predictive_model, self.epm_optimizers
        ):
            predictive_network.train()
            # Get the deterministic prediction of each model
            prediction_vector = predictive_network(latent_state, actions)
            # Calculate Loss of each model
            loss = F.mse_loss(prediction_vector, latent_next_state)
            # Update weights and bias
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> None:
        self.encoder.train()
        self.decoder.train()
        self.actor.train()
        self.critic.train()

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

        # Normalise states and next_states
        # This because the states are [0-255] and the predictions are [0-1]
        states /= 255
        next_states /= 255

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # Update the Critic
        self._update_critic(states, actions, rewards, next_states, dones)

        # Update Autoencoder
        self._update_autoencoder(states)

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            self._update_actor(states)

            # Update target network params
            # Note: the encoders in target networks are the same of main networks, so I wont update them
            hlp.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.tau)
            hlp.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.tau)

            hlp.soft_update_params(
                self.actor.act_net, self.actor_target.act_net, self.tau
            )

        # Update intrinsic models
        if self.intrinsic_on:
            self._update_predictive_model(states, actions, next_states)

    def get_intrinsic_reward(
        self, state: np.ndarray, action: np.ndarray, next_state: np.ndarray
    ) -> float:
        with torch.no_grad():
            # Normalise states and next_states
            # This because the states are [0-255] and the predictions are [0-1]
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            state_tensor = state_tensor / 255

            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            next_state_tensor = next_state_tensor.unsqueeze(0)
            next_state_tensor = next_state_tensor / 255

            action_tensor = torch.FloatTensor(action).to(self.device)
            action_tensor = action_tensor.unsqueeze(0)

            surprise_rate = self._get_surprise_rate(
                state_tensor, action_tensor, next_state_tensor
            )
            novelty_rate = self._get_novelty_rate(state_tensor)

        # TODO make these parameters - i.e. Tony's work
        a = 1.0
        b = 1.0
        reward_surprise = surprise_rate * a
        reward_novelty = novelty_rate * b

        return reward_surprise + reward_novelty

    def _get_surprise_rate(
        self,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        next_state_tensor: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            latent_state = self.encoder(state_tensor, detach_output=True)
            latent_next_state = self.encoder(next_state_tensor, detach_output=True)

            predict_vector_set = []
            for network in self.ensemble_predictive_model:
                network.eval()
                predicted_vector = network(latent_state, action_tensor)
                predict_vector_set.append(predicted_vector.detach().cpu().numpy())
            ensemble_vector = np.concatenate(predict_vector_set, axis=0)
            z_next_latent_prediction = np.mean(ensemble_vector, axis=0)
            z_next_latent_true = latent_next_state.detach().cpu().numpy()[0]
            mse = (np.square(z_next_latent_prediction - z_next_latent_true)).mean()
        return mse

    def _get_novelty_rate(self, state_tensor_img: torch.Tensor) -> float:
        with torch.no_grad():
            z_vector = self.encoder(state_tensor_img, detach_output=True)

            # rec_img is a stack of k images --> (1, k , 84 ,84), [0~1]
            rec_img = self.decoder(z_vector)

            # --> (k , 84 ,84)
            original_stack_imgs = state_tensor_img.cpu().numpy()[0]
            reconstruction_stack = rec_img.cpu().numpy()[0]

        target_images = original_stack_imgs
        ssim_index_total = ssim(
            target_images,
            reconstruction_stack,
            full=False,
            data_range=target_images.max() - target_images.min(),
            channel_axis=0,
        )
        novelty_rate = 1 - ssim_index_total

        return novelty_rate

    def _get_reconstruction_for_evaluation(self, state: np.ndarray) -> np.ndarray:
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            state_tensor_img = torch.FloatTensor(state).to(self.device)
            state_tensor_img = state_tensor_img.unsqueeze(0)
            z_vector = self.encoder(state_tensor_img)
            rec_img = self.decoder(z_vector)
            rec_img = rec_img.cpu().numpy()[0]  # --> (k , 84 ,84)

        original_img = np.moveaxis(state, 0, -1)  # --> (84 ,84, 3)
        original_img = np.array_split(original_img, 3, axis=2)
        rec_img = np.moveaxis(rec_img, 0, -1)
        rec_img = np.array_split(rec_img, 3, axis=2)

        self.encoder.train()
        self.decoder.train()

        return original_img, rec_img

    def save_models(self, filename: str, filepath: str = "models") -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor.state_dict(), f"{path}/{filename}_actor.pht")
        torch.save(self.critic.state_dict(), f"{path}/{filename}_critic.pht")
        torch.save(self.encoder.state_dict(), f"{path}/{filename}_encoder.pht")
        torch.save(self.decoder.state_dict(), f"{path}/{filename}_decoder.pht")
        torch.save(
            self.ensemble_predictive_model.state_dict(),
            f"{path}/{filename}_ensemble.pht",
        )
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath
        self.actor.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        self.encoder.load_state_dict(torch.load(f"{path}/{filename}_encoder.pht"))
        self.decoder.load_state_dict(torch.load(f"{path}/{filename}_decoder.pht"))
        logging.info("models has been loaded...")
