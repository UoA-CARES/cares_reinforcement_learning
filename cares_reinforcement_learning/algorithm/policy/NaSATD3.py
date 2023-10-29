
"""
NaSA-TD3: Novelty and Surprise Autoencoder TD3
Learning from Images
Algorithm for Gym and DM Control Suite
"""

import os
import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import numpy as np
from skimage.metrics import structural_similarity as ssim # This is used to metric the novelty.

from cares_reinforcement_learning.networks.NaSATD3.EPDM import EPDM # TODO no sure how to import this, the ensemble will be the same? Can I pass this form outside?

class NaSATD3:
    def __init__(self,
                 encoder_network,
                 decoder_network,
                 actor_network,
                 critic_network,
                 gamma,
                 tau,
                 action_num,
                 latent_size,
                 intrinsic_on,
                 device):

        self.type = "policy"
        self.gamma = gamma
        self.tau   = tau
        self.ensemble_size = 3 # TODO move to parameters?
        self.latent_size   = latent_size
        self.intrinsic_on  = intrinsic_on

        self.learn_counter      = 0
        self.policy_update_freq = 2

        self.action_num  = action_num
        self.device      = device

        self.encoder = encoder_network.to(device)
        self.decoder = decoder_network.to(device)
        self.actor   = actor_network.to(device)
        self.critic  = critic_network.to(device)

        self.actor_target  = copy.deepcopy(self.actor).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Necessary for make the same encoder in the whole algorithm
        self.actor_target.encoder_net  = self.encoder
        self.critic_target.encoder_net = self.encoder

        self.epm = nn.ModuleList()
        networks = [EPDM(self.latent_size, self.action_num) for _ in range(self.ensemble_size)]
        self.epm.extend(networks)
        self.epm.to(self.device)

        # TODO move these to parameters through NaSATD3Config
        lr_actor   = 1e-4
        lr_critic  = 1e-3
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  lr=lr_critic)

        lr_encoder = 1e-4
        lr_decoder = 1e-4
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr_encoder)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr_decoder, weight_decay=1e-7)

        lr_epm      = 1e-4
        w_decay_epm = 1e-3
        self.epm_optimizers = [torch.optim.Adam(self.epm[i].parameters(), lr=lr_epm, weight_decay=w_decay_epm) for i in range(self.ensemble_size)]


    def select_action_from_policy(self, state, evaluation=False, noise_scale=0.1):
        self.actor.eval()
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).to(self.device)
            state_tensor = state_tensor.unsqueeze(0)
            action = self.actor(state_tensor)
            action = action.cpu().data.numpy().flatten()
            if not evaluation:
                # this is part the TD3 too, add noise to the action
                noise = np.random.normal(0, scale=noise_scale, size=self.action_num)
                action = action + noise
                action = np.clip(action, -1, 1)
        self.actor.train()
        return action


    def train_policy(self, experiences):
        self.encoder.train()
        self.decoder.train()
        self.actor.train()
        self.critic.train()

        self.learn_counter += 1

        # Note: for gripper the goal angle need to be passed here as well
        states, actions, rewards, next_states, dones = experiences
        batch_size = len(states)

        # Convert into tensor
        states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        rewards     = torch.FloatTensor(np.asarray(rewards)).to(self.device)
        next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)
        dones       = torch.LongTensor(np.asarray(dones)).to(self.device)
        # goals       = torch.FloatTensor(np.asarray(goals)).to(self.device)

        # Reshape to batch_size
        rewards = rewards.unsqueeze(0).reshape(batch_size, 1)
        dones   = dones.unsqueeze(0).reshape(batch_size, 1)

        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_noise = 0.2 * torch.randn_like(next_actions)
            target_noise = torch.clamp(target_noise, -0.5, 0.5)
            next_actions = next_actions + target_noise
            next_actions = torch.clamp(next_actions, min=-1, max=1)

            target_q_values_one, target_q_values_two = self.critic_target(next_states, next_actions)
            target_q_values = torch.minimum(target_q_values_one, target_q_values_two)

            q_target = rewards + self.gamma * (1 - dones) * target_q_values

        q_values_one, q_values_two = self.critic(states, actions)

        critic_loss_1 = F.mse_loss(q_values_one, q_target)
        critic_loss_2 = F.mse_loss(q_values_two, q_target)
        critic_loss_total = critic_loss_1 + critic_loss_2

        # Update the Critic
        self.critic_optimizer.zero_grad()
        critic_loss_total.backward()
        self.critic_optimizer.step()

        # Update Autoencoder
        z_vector = self.encoder(states)
        rec_obs  = self.decoder(z_vector)

        target_images = states / 255 # this because the states are [0-255] and the predictions are [0-1]
        rec_loss      = F.mse_loss(target_images, rec_obs)

        latent_loss = (0.5 * z_vector.pow(2).sum(1)).mean()  # add L2 penalty on latent representation
        ae_loss     = rec_loss + 1e-6 * latent_loss

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # Update Actor
        if self.learn_counter % self.policy_update_freq == 0:
            actor_q_one, actor_q_two = self.critic(states, self.actor(states, detach_encoder=True),  detach_encoder=True)
            actor_q_values           = torch.minimum(actor_q_one, actor_q_two)
            actor_loss               = -actor_q_values.mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target network params
            for target_param, param in zip(self.critic_target.Q1.parameters(), self.critic.Q1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic_target.Q2.parameters(), self.critic.Q2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.actor_target.act_net.parameters(), self.actor.act_net.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            # Note: the encoders in target networks are the same of main networks, so I wont update them
        
        # TODO split the tuple non-tuple
        # Update intrinsic models
        if self.intrinsic_on:
            self.train_predictive_model(states, actions, next_states)

    def get_intrinsic_reward(self, state, action, next_state):
        with torch.no_grad():
            state_tensor      = torch.FloatTensor(state).to(self.device)
            state_tensor      = state_tensor.unsqueeze(0)
            next_state_tensor = torch.FloatTensor(next_state).to(self.device)
            next_state_tensor = next_state_tensor.unsqueeze(0)
            action_tensor     = torch.FloatTensor(action).to(self.device)
            action_tensor     = action_tensor.unsqueeze(0)

            surprise_rate = self.get_surprise_rate(state_tensor, action_tensor, next_state_tensor)
            novelty_rate  = self.get_novelty_rate(state_tensor)

        #TODO make these parameters - i.e. Tony's work
        a = 1.0
        b = 1.0
        reward_surprise = surprise_rate * a
        reward_novelty  = novelty_rate  * b

        return reward_surprise + reward_novelty

    def get_surprise_rate(self, state_tensor, action_tensor, next_state_tensor):
        with torch.no_grad():
            latent_state      = self.encoder(state_tensor, detach=True)
            latent_next_state = self.encoder(next_state_tensor, detach=True)

            predict_vector_set = []
            for network in self.epm:
                network.eval()
                predicted_vector = network(latent_state, action_tensor)
                predict_vector_set.append(predicted_vector.detach().cpu().numpy())
            ensemble_vector = np.concatenate(predict_vector_set, axis=0)
            z_next_latent_prediction = np.mean(ensemble_vector, axis=0)
            z_next_latent_true       = latent_next_state.detach().cpu().numpy()[0]
            mse = (np.square(z_next_latent_prediction - z_next_latent_true)).mean()
        return mse

    def get_novelty_rate(self, state_tensor_img):
        with torch.no_grad():
            z_vector = self.encoder(state_tensor_img)
            rec_img  = self.decoder(z_vector)  # rec_img is a stack of k images --> (1, k , 84 ,84), [0~255]

            original_stack_imgs  = state_tensor_img.cpu().numpy()[0]  # --> (k , 84 ,84)
            reconstruction_stack = rec_img.cpu().numpy()[0]           # --> (k , 84 ,84)

        target_images    = original_stack_imgs / 255
        ssim_index_total = ssim(target_images, reconstruction_stack, full=False, data_range=target_images.max() - target_images.min(), channel_axis=0)
        novelty_rate     = 1 - ssim_index_total

        return novelty_rate

    def train_predictive_model(self, states, actions, next_states):
        # states, actions, next_states = experiences

        # states      = torch.FloatTensor(np.asarray(states)).to(self.device)
        # actions     = torch.FloatTensor(np.asarray(actions)).to(self.device)
        # next_states = torch.FloatTensor(np.asarray(next_states)).to(self.device)

        with torch.no_grad():
            latent_state      = self.encoder(states, detach=True)
            latent_next_state = self.encoder(next_states, detach=True)

        for predictive_network, optimizer in zip(self.epm, self.epm_optimizers):
            predictive_network.train()
            # Get the deterministic prediction of each model
            prediction_vector = predictive_network(latent_state, actions)
            # Calculate Loss of each model
            loss = F.mse_loss(prediction_vector, latent_next_state)
            # Update weights and bias
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def get_reconstruction_for_evaluation(self, state):
        self.encoder.eval()
        self.decoder.eval()
        with torch.no_grad():
            state_tensor_img = torch.FloatTensor(state).to(self.device)
            state_tensor_img = state_tensor_img.unsqueeze(0)
            z_vector = self.encoder(state_tensor_img)
            rec_img  = self.decoder(z_vector)
            rec_img  = rec_img.cpu().numpy()[0]  # --> (k , 84 ,84)

        original_img = np.moveaxis(state, 0, -1)  # --> (84 ,84, 3)
        original_img = np.array_split(original_img, 3, axis=2)
        rec_img      = np.moveaxis(rec_img, 0, -1)
        rec_img      = np.array_split(rec_img, 3, axis=2)

        self.encoder.train()
        self.decoder.train()

        return original_img, rec_img

    def save_models(self, filename, filepath='models'):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        dir_exists = os.path.exists(path)

        if not dir_exists:
            os.makedirs(path)
        torch.save(self.actor.state_dict(),   f'{path}/{filename}_actor.pht')
        torch.save(self.critic.state_dict(),  f'{path}/{filename}_critic.pht')
        torch.save(self.encoder.state_dict(), f'{path}/{filename}_encoder.pht')
        torch.save(self.decoder.state_dict(), f'{path}/{filename}_decoder.pht')
        torch.save(self.epm.state_dict(),     f'{path}/{filename}_ensemble.pht')
        logging.info("models has been saved...")

    def load_models(self, filepath, filename):
        path = f"{filepath}/models" if filepath != 'models' else filepath
        self.actor.load_state_dict(torch.load(f'{path}/{filename}_actor.pht'))
        self.critic.load_state_dict(torch.load(f'{path}/{filename}_critic.pht'))
        self.encoder.load_state_dict(torch.load(f'{path}/{filename}_encoder.pht'))
        self.decoder.load_state_dict(torch.load(f'{path}/{filename}_decoder.pht'))
        logging.info("models has been loaded...")
