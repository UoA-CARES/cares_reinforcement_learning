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
from cares_reinforcement_learning.encoders.configurations import VanillaAEConfig
from cares_reinforcement_learning.encoders.losses import AELoss
from cares_reinforcement_learning.encoders.types import AECompositeState
from cares_reinforcement_learning.memory import MemoryBuffer
from typing import List


class TD3AE:
    def __init__(
        self,
        actor_network: torch.nn.Module,
        critic_network: torch.nn.Module,
        decoder_network: torch.nn.Module,
        gamma: float,
        tau: float,
        action_num: int,
        actor_lr: float,
        critic_lr: float,
        encoder_tau: float,
        decoder_update_freq: int,
        ae_config: VanillaAEConfig,
        device: torch.device,
    ):
        self.type = "policy"
        self.device = device

        self.actor_net = actor_network.to(self.device)
        self.critic_net = critic_network.to(self.device)

        self.target_actor_net = copy.deepcopy(self.actor_net)
        self.target_critic_net = copy.deepcopy(self.critic_net)

        # tie the encoder weights
        self.actor_net.encoder.copy_conv_weights_from(self.critic_net.encoder)

        self.encoder_tau = encoder_tau

        self.decoder_net = decoder_network.to(device)
        self.decoder_latent_lambda = ae_config.latent_lambda
        self.decoder_update_freq = decoder_update_freq

        self.gamma = gamma
        self.tau = tau

        self.noise_clip = 0.5
        self.policy_noise = 0.2

        self.learn_counter = 0
        self.policy_update_freq = 2

        self.action_num = action_num

        self.actor_net_optimiser = torch.optim.Adam(
            self.actor_net.parameters(), lr=actor_lr
        )
        self.critic_net_optimiser = torch.optim.Adam(
            self.critic_net.parameters(), lr=critic_lr
        )

        self.loss_function = AELoss(latent_lambda=ae_config.latent_lambda)

        self.encoder_net_optimiser = torch.optim.Adam(
            self.critic_net.encoder.parameters(), **ae_config.encoder_optim_kwargs
        )
        self.decoder_net_optimiser = torch.optim.Adam(
            self.decoder_net.parameters(),
            **ae_config.decoder_optim_kwargs,
        )
        # needed since tensor shapes need to be treated differently, read this info from autoencoder config
        self.is_1d = ae_config.is_1d

    def select_action_from_policy(
        self, state: dict, evaluation: bool = False, noise_scale: float = 0.1
    ) -> np.ndarray:
        
        # NOT TENSORS YET, JUST ARRAYS / NP ARRAYS
        images = state['image']
        info = state["vector"]

        self.actor_net.eval()
        with torch.no_grad():
            # state_tensor = torch.FloatTensor(state).to(self.device)


            ## TODO: Doesn't make sense to normalize for non-image input this way, but not breaking
            image_tensor = torch.FloatTensor(images).to(self.device)/255

            # all modules expect batched input, state pulled straight from source (not a sampler) does not have batch dim, thus fixing shapes here
            if self.is_1d:
                correct_batched_dim = 3 # batch,channel,length
            else:
                correct_batched_dim = 4 # batch,channel,width,height
            
            # Using a loop since torch.Tensor([1,2,3]) or np.array([1,2,3]) yield shape of (3,) and NOT (1,3), but [[1,2,3],[4,5,6]] have shape (2,3)
            # both are 1d cases but one need to be unsqueezed twice instead of once, so might aswell generalize it
            while image_tensor.dim() < correct_batched_dim:
                image_tensor = image_tensor.unsqueeze(0)

           
            info_tensor = torch.FloatTensor(info).to(self.device)
            # let length be L: (L,) -> (1,L)
            while info_tensor.dim() < 2:
                info_tensor = info_tensor.unsqueeze(0)
            
            composite_state:AECompositeState = {
                'image': image_tensor,
                'vector': info_tensor
            }

           
            # if self.is_1d:
            #     state_tensor = state_tensor.unsqueeze(0).unsqueeze(0)
            # else:
            #     state_tensor = state_tensor.unsqueeze(0)

            # TODO: Doesn't make sense for non-image input, but not breaking
            # state_tensor = state_tensor / 255

            action = self.actor_net(composite_state)
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
        states: AECompositeState,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        next_states: AECompositeState,
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

    def _update_actor(self, states: AECompositeState) -> float:
        actions = self.actor_net(states, detach_encoder=True)
        actor_q_values, _ = self.critic_net(states, actions, detach_encoder=True)
        actor_loss = -actor_q_values.mean()

        self.actor_net_optimiser.zero_grad()
        actor_loss.backward()
        self.actor_net_optimiser.step()

        return actor_loss.item()

    def _update_autoencoder(self, states: AECompositeState) -> float:
        
        image = states['image']

        latent_samples = self.critic_net.encoder(image)

        reconstructed_data = self.decoder_net(latent_samples)

        ae_loss = self.loss_function.calculate_loss(
            data=image,
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
        
        '''
        states: 
        [ 
            { 
                'image': np.array, shape: (channels * dim1 * dim2), 1D case: (channels * length)
                'vector': np.array, shape: (length)
            } ...
        ]
        CHANNELS INCLUDE STACK SIZE FOR TEMPORAL INFO. e.g. a stack of 3 RGB images have 9 channels
        '''

        # Convert into tensor
        # TODO: can probably be optimized by using a np array with known size
        image_arr = []
        vector_arr = []
        for state in states:
            image_arr.append(state["image"])
            vector_arr.append(state["vector"])
        
        states:AECompositeState = {
            'image': torch.FloatTensor(np.array(image_arr)).to(self.device),
            'vector': torch.FloatTensor(np.array(vector_arr)).to(self.device)
        }
        # states = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards = torch.FloatTensor(np.asarray(rewards)).to(self.device)

        # TODO: same here
        next_image_arr = []
        next_vector_arr = []
        for next_state in next_states:
            next_image_arr.append(next_state["image"])
            next_vector_arr.append(next_state["vector"])
        next_states:AECompositeState = {
            'image': torch.FloatTensor(np.array(next_image_arr)).to(self.device),
            'vector': torch.FloatTensor(np.array(next_vector_arr)).to(self.device)
        }
        # next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones = torch.LongTensor(np.asarray(dones)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones = dones.unsqueeze(0).reshape(batch_size, 1)

        # Here since passing states directly in result in shape [1,batch_size,num_of_features] SOMEHOW
        # might be related to that weird omitting size of 1 issue
        # if self.is_1d:
        #     states = states.view(batch_size, 1, -1)
        #     next_states = next_states.view(batch_size, 1, -1)

        # TODO: does not make sense for non-image cases. However some scaling does not break anything either.
        # Normalise states and next_states
        # This because the states are [0-255] and the predictions are [0-1]
        states_normalised:AECompositeState = {
            'image': states['image']/255,
            'vector': states["vector"]
        }

        next_states_normalised:AECompositeState = {
            'image': next_states['image']/255,
            'vector': next_states['vector']
        }

        info = {}

        critic_loss_one, critic_loss_two, critic_loss_total = self._update_critic(
            states_normalised, actions, rewards, next_states_normalised, dones
        )
        info["critic_loss_one"] = critic_loss_one
        info["critic_loss_two"] = critic_loss_two
        info["critic_loss"] = critic_loss_total

        if self.learn_counter % self.policy_update_freq == 0:
            # Update Actor
            actor_loss = self._update_actor(states_normalised)
            info["actor_loss"] = actor_loss

            # Update target network params
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

            hlp.soft_update_params(
                self.actor_net.act_net, self.target_actor_net.act_net, self.encoder_tau
            )

            hlp.soft_update_params(
                self.actor_net.encoder, self.target_actor_net.encoder, self.encoder_tau
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

        logging.info("models has been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        path = f"{filepath}/models" if filepath != "models" else filepath

        self.actor_net.load_state_dict(torch.load(f"{path}/{filename}_actor.pht"))
        self.critic_net.load_state_dict(torch.load(f"{path}/{filename}_critic.pht"))
        logging.info("models has been loaded...")
