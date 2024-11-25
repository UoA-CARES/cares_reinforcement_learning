import copy
import logging
import os
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

# This is used to metric the novelty.
from skimage.metrics import structural_similarity as ssim
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.vanilla_autoencoder import VanillaAutoencoder
from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.networks.NaSATD3 import Actor, Critic
from cares_reinforcement_learning.networks.NaSATD3.EPDM import EPDM
from cares_reinforcement_learning.util.configurations import NaSATD3Config


class NaSATD3:
    def __init__(
        self,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        actor_network: Actor,
        critic_network: Critic,
        config: NaSATD3Config,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        self.gamma = config.gamma
        self.tau = config.tau

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.ensemble_size = config.ensemble_size
        self.intrinsic_on = config.intrinsic_on

        self.learn_counter = 0
        self.policy_update_freq = config.policy_update_freq

        self.autoencoder = autoencoder.to(device)

        self.actor = actor_network.to(device)
        self.critic = critic_network.to(device)

        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)

        self.action_num = self.actor.num_actions

        # Necessary to make the same autoencoder in the whole algorithm
        self.actor_target.autoencoder = self.autoencoder
        self.critic_target.autoencoder = self.autoencoder

        self.ensemble_predictive_model = nn.ModuleList()
        networks = [
            EPDM(
                self.autoencoder.latent_dim,
                self.action_num,
                hidden_size=config.hidden_size_epdm,
            )
            for _ in range(self.ensemble_size)
        ]
        self.ensemble_predictive_model.extend(networks)
        self.ensemble_predictive_model.to(self.device)

        self.actor_lr = config.actor_lr
        self.critic_lr = config.critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.critic_lr
        )

        self.epm_lr = config.epm_lr
        self.epm_optimizers = [
            torch.optim.Adam(
                self.ensemble_predictive_model[i].parameters(),
                lr=self.epm_lr,
                weight_decay=1e-3,
            )
            for i in range(self.ensemble_size)
        ]

    def select_action_from_policy(
        self,
        state: dict[str, np.ndarray],
        evaluation: bool = False,
        noise_scale: float = 0.1,
    ) -> np.ndarray:
        self.actor.eval()
        self.autoencoder.eval()
        with torch.no_grad():
            state_tensor = hlp.image_state_dict_to_tensor(state, self.device)

            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                action += noise_scale * np.random.randn(self.action_num)
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)

        self.actor.train()
        self.autoencoder.train()
        return action

    def _update_critic(
        self,
        states: dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: torch.Tensor,
        dones: torch.Tensor,
    ) -> tuple[float, float, float]:
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

        return critic_loss_one.item(), critic_loss_two.item(), critic_loss_total.item()

    def _update_autoencoder(self, states: torch.Tensor) -> float:
        # Leaving this function in case this needs to be extended again in the future
        ae_loss = self.autoencoder.update_autoencoder(states)
        return ae_loss.item()

    def _update_actor(self, states: dict[str, torch.Tensor]) -> float:
        actor_q_one, actor_q_two = self.critic(
            states, self.actor(states, detach_encoder=True), detach_encoder=True
        )
        actor_q_values = torch.minimum(actor_q_one, actor_q_two)
        actor_loss = -actor_q_values.mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return actor_loss.item()

    def _get_latent_state(
        self, states: np.ndarray, detach_output: bool, sample_latent: bool = True
    ) -> torch.Tensor:
        # NaSATD3 detatches the encoder at the output
        output = self.autoencoder.encoder(states, detach_output=detach_output)
        latent_state = output

        if self.autoencoder.ae_type == Autoencoders.BURGESS:
            latent_state, _, _ = output
            # take the sample value for the latent space
            if sample_latent:
                _, _, latent_state = output

        return latent_state

    def _update_predictive_model(
        self,
        states: dict[str, torch.Tensor],
        actions: np.ndarray,
        next_states: dict[str, torch.Tensor],
    ) -> list[float]:

        with torch.no_grad():
            latent_state = self._get_latent_state(
                states["image"], detach_output=True, sample_latent=True
            )

            latent_next_state = self._get_latent_state(
                next_states["image"], detach_output=True, sample_latent=True
            )

        pred_losses = []
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

            pred_losses.append(loss.item())

        return pred_losses

    def train_policy(self, memory: MemoryBuffer, batch_size: int) -> dict[str, Any]:
        self.actor.train()
        self.critic.train()
        self.autoencoder.train()
        self.autoencoder.encoder.train()
        self.autoencoder.decoder.train()

        self.learn_counter += 1

        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, _ = experiences

        batch_size = len(states)

        states = hlp.image_states_dict_to_tensor(states, self.device)

        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)

        next_states = hlp.image_states_dict_to_tensor(next_states, self.device)

        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        info = {}

        # Update the Critic
        critic_loss_one, critic_loss_two, critic_loss_total = self._update_critic(
            states, actions, rewards, next_states, dones
        )
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["critic_loss_total"] = critic_loss_total

        # Update Autoencoder
        ae_loss = self._update_autoencoder(states["image"])
        info["ae_loss"] = ae_loss

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_loss = self._update_actor(states)
            info["actor_loss"] = actor_loss

            # Update target network params
            # Note: the encoders in target networks are the same of main networks, so I wont update them
            hlp.soft_update_params(self.critic.Q1, self.critic_target.Q1, self.tau)
            hlp.soft_update_params(self.critic.Q2, self.critic_target.Q2, self.tau)

            hlp.soft_update_params(
                self.actor.act_net, self.actor_target.act_net, self.tau
            )

        # Update intrinsic models
        if self.intrinsic_on:
            pred_losses = self._update_predictive_model(states, actions, next_states)
            info["pred_losses"] = pred_losses

        return info

    def _get_surprise_rate(
        self,
        state_tensor: torch.Tensor,
        action_tensor: torch.Tensor,
        next_state_tensor: torch.Tensor,
    ) -> float:
        with torch.no_grad():
            latent_state = self._get_latent_state(
                state_tensor, detach_output=True, sample_latent=True
            )

            latent_next_state = self._get_latent_state(
                next_state_tensor, detach_output=True, sample_latent=True
            )

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
            # rec_img is a stack of k images --> (1, k , w ,w), [0~1]
            output = self.autoencoder(state_tensor_img, detach_output=True)
            rec_img = output["reconstructed_observation"]

            # --> (k , w , w)
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

    def get_intrinsic_reward(
        self,
        state: dict[str, np.ndarray],
        action: np.ndarray,
        next_state: dict[str, np.ndarray],
    ) -> float:
        with torch.no_grad():
            vector_tensor = torch.FloatTensor(state["vector"])
            vector_tensor = vector_tensor.unsqueeze(0).to(self.device)

            image_tensor = torch.FloatTensor(state["image"])
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            image_tensor = image_tensor / 255

            state_tensor = {"image": image_tensor, "vector": vector_tensor}

            next_vector_tensor = torch.FloatTensor(next_state["vector"])
            next_vector_tensor = vector_tensor.unsqueeze(0).to(self.device)

            next_image_tensor = torch.FloatTensor(next_state["image"])
            next_image_tensor = next_image_tensor.unsqueeze(0).to(self.device)
            next_image_tensor = next_image_tensor / 255

            next_state_tensor = {
                "image": next_image_tensor,
                "vector": next_vector_tensor,
            }

            action_tensor = torch.FloatTensor(action).to(self.device)
            action_tensor = action_tensor.unsqueeze(0)

            surprise_rate = self._get_surprise_rate(
                state_tensor["image"], action_tensor, next_state_tensor["image"]
            )
            novelty_rate = self._get_novelty_rate(state_tensor["image"])

        # TODO make these parameters - i.e. Tony's work
        a = 1.0
        b = 1.0
        reward_surprise = surprise_rate * a
        reward_novelty = novelty_rate * b

        return reward_surprise + reward_novelty

    # def _get_reconstruction_for_evaluation(self, state: np.ndarray) -> np.ndarray:
    #     self.encoder.eval()
    #     self.decoder.eval()
    #     with torch.no_grad():
    #         state_tensor_img = torch.FloatTensor(state).to(self.device)
    #         state_tensor_img = state_tensor_img.unsqueeze(0)
    #         z_vector = self.encoder(state_tensor_img)
    #         rec_img = self.decoder(z_vector)
    #         rec_img = rec_img.cpu().numpy()[0]  # --> (k , 84 ,84)

    #     original_img = np.moveaxis(state, 0, -1)  # --> (84 ,84, 3)
    #     original_img = np.array_split(original_img, 3, axis=2)
    #     rec_img = np.moveaxis(rec_img, 0, -1)
    #     rec_img = np.array_split(rec_img, 3, axis=2)

    #     self.encoder.train()
    #     self.decoder.train()

    #     return original_img, rec_img

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        torch.save(self.actor.state_dict(), f"{filepath}/{filename}_actor.pht")
        torch.save(self.critic.state_dict(), f"{filepath}/{filename}_critic.pht")
        torch.save(
            self.autoencoder.encoder.state_dict(), f"{filepath}/{filename}_encoder.pht"
        )
        torch.save(
            self.autoencoder.decoder.state_dict(), f"{filepath}/{filename}_decoder.pht"
        )
        torch.save(
            self.ensemble_predictive_model.state_dict(),
            f"{filepath}/{filename}_ensemble.pht",
        )
        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        self.actor.load_state_dict(torch.load(f"{filepath}/{filename}_actor.pht"))
        self.critic.load_state_dict(torch.load(f"{filepath}/{filename}_critic.pht"))
        self.autoencoder.encoder.load_state_dict(
            torch.load(f"{filepath}/{filename}_encoder.pht")
        )
        self.autoencoder.decoder.load_state_dict(
            torch.load(f"{filepath}/{filename}_decoder.pht")
        )
        logging.info("models has been loaded...")
