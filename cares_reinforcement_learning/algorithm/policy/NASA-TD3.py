
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

    #def __init__(self, latent_size, action_num, device, k):
    def __init__(self, actor_network, critic_network, encoder_net, decoder_net, action_num, device, k):

        self.latent_size = latent_size
        self.action_num  = action_num
        self.device      = device


        # self.k = k * 3 # number of stack frames*3, 3 because I am using 3CH images
        #
        # self.encoder = Encoder(latent_dim=self.latent_size, k=self.k).to(self.device)
        # self.decoder = Decoder(latent_dim=self.latent_size, k=self.k).to(self.device)
        #
        # # TODO note that for gripper I need to inject the goal angle so the laten_size become latent_size+1
        # self.actor  = Actor(self.latent_size, self.action_num, self.encoder).to(self.device)
        # self.critic = Critic(self.latent_size, self.action_num, self.encoder).to(self.device)
        #
        # # TODO test what happen with copy.deepcopy(self.actor_net).to(device)
        # self.actor_target  = Actor(self.latent_size, self.action_num, self.encoder).to(self.device)
        # self.critic_target = Critic(self.latent_size, self.action_num, self.encoder).to(self.device)
        #
        # self.critic_target.load_state_dict(self.critic.state_dict())
        # self.actor_target.load_state_dict(self.actor.state_dict())
        #
        # self.epm = nn.ModuleList()
        # networks = [EPDM(self.latent_size, self.action_num) for _ in range(self.ensemble_size)]
        # self.epm.extend(networks)
        # self.epm.to(self.device)

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

        self.gamma = 0.99
        self.tau   = 0.005
        self.ensemble_size = 10

        self.learn_counter      = 0
        self.policy_update_freq = 2

