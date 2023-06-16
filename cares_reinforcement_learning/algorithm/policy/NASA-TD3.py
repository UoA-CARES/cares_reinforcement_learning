
"""
NaSA-TD3: Novelty and Surprise Autoencoder TD3
"""

import os
import logging
logging.basicConfig(level=logging.INFO)

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from skimage.metrics import structural_similarity as ssim # This is used to metric the novelty.

from networks import EPDM  # Deterministic Ensemble


class NASA_TD3:

    def __init__(self,
                 encoder_network,
                 decoder_network,
                 actor_network,
                 critic_network,
                 action_num,
                 latent_size,
                 device):

        self.gamma = 0.99
        self.tau   = 0.005
        self.ensemble_size = 10
        self.latent_size   = latent_size

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
        self.actor_target.encoder_net = self.encoder
        self.critic_target.encoder_net = self.encoder

        self.epm = nn.ModuleList()
        networks = [EPDM(self.latent_size, self.action_num) for _ in range(self.ensemble_size)]
        self.epm.extend(networks)
        self.epm.to(self.device)

        lr_actor   = 1e-4
        lr_critic  = 1e-3
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),   lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),  lr=lr_critic)

        lr_encoder = 1e-3
        lr_decoder = 1e-3
        self.encoder_optimizer = torch.optim.Adam(self.encoder.parameters(), lr=lr_encoder)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=lr_decoder, weight_decay=1e-7)

        lr_epm = 1e-4
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

