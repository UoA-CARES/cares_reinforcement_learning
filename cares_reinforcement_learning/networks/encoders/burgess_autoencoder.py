"""

Original Code: https://github.com/YannDubs/disentangling-vae/tree/master
"""

import numpy as np
import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp


class BurgessAutoencoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
    ):
        super().__init__()

        self.encoder = BurgessEncoder(
            observation_size,
            latent_dim,
            num_layers,
            num_filters,
            kernel_size,
        )
        self.decoder = BurgessDecoder(
            observation_size,
            latent_dim,
            self.encoder.out_dim,
            num_layers,
            num_filters,
            kernel_size,
        )

        # self.apply(weight_init_burgess)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def forward(
        self,
        states,
        storer=None,
        n_data=1000,
        detach=False,
        no_discrim=False,
        is_agent=False,
    ):
        # states = states / 255

        mean, logvar = self.encoder(states, detach)
        print(f"{mean.shape=}")

        latent_sample = self.reparameterize(mean, logvar)
        print(f"{latent_sample.shape=}")

        rec_obs = self.decoder(latent_sample)
        print(f"{rec_obs.shape=}")
        loss = 0

        # loss = self.loss_fn(
        #     states,
        #     rec_obs,
        #     latent_dist,
        #     True,
        #     storer,
        #     latent_sample=latent_sample,
        #     n_data=n_data,
        #     no_discrim=no_discrim,
        #     is_agent=is_agent,
        # )

        if detach:
            latent_sample = latent_sample.detach()

        return {
            "loss": loss,
            "z_vector": latent_sample,
            "latent_dist": [mean, logvar],
            "rec_obs": rec_obs,
        }

    def sample_latent(self, x):
        latent_dist = self.encoder(x)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class BurgessEncoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        stride = 2
        padding = 0
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    observation_size[0],
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=padding,
                )
            ]
        )

        self.out_dim = hlp.flatten(
            observation_size[1], k=self.kernel_size, s=stride, p=padding
        )

        stride = 1
        padding = 0
        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv2d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.out_dim = hlp.flatten(
                self.out_dim, k=self.kernel_size, s=stride, p=padding
            )

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        # Fully connected layers
        self.linear_one = nn.Linear(self.n_flatten, self.hidden_dim)
        self.linear_two = nn.Linear(self.hidden_dim, self.hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(self.hidden_dim, self.latent_dim * 2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        conv = torch.relu(self.convs[0](x))
        print(f"{conv.shape=}")

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            print(f"{conv.shape=}")

        return conv

    def enc_forward(
        self, obs: torch.Tensor, detach_cnn: bool = False, detach_output: bool = False
    ):
        batch_size = obs.size(0)

        # Convolutional layers with ReLu activations
        print(f"{obs.shape=}")
        h = self._forward_conv(obs)

        # SAC AE detaches at the CNN layer
        if detach_cnn:
            h = h.detach()

        # Fully connected layers with ReLu activations
        h = h.view((batch_size, -1))
        print(f"{h.shape=}")
        h = torch.relu(self.linear_one(h))
        print(f"{h.shape=}")
        h = torch.relu(self.linear_two(h))
        print(f"{h.shape=}")

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(h)
        print(f"{mu_logvar.shape=}")

        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)
        print(f"{mu.shape=}")
        print(f"{logvar.shape=}")

        if detach_output:
            mu = mu.detach()
            logvar = logvar.detach()

        return mu, logvar

    def forward(self, x, detach=False):
        latent_dist = self.enc_forward(x, detach)
        return latent_dist


class BurgessDecoder(nn.Module):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        out_dim: int,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 4,
        hidden_dim: int = 256,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.out_dim = out_dim

        self.num_layers = num_layers
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.hidden_dim = hidden_dim

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        # Fully connected layers
        self.linear_one = nn.Linear(latent_dim, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, hidden_dim)
        self.linear_three = nn.Linear(hidden_dim, self.n_flatten)

        self.deconvs = nn.ModuleList()

        stride = 1
        padding = 0
        for _ in range(self.num_layers - 1):
            self.deconvs.append(
                nn.ConvTranspose2d(
                    in_channels=self.num_filters,
                    out_channels=self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )

        stride = 2
        padding = 1
        self.deconvs.append(
            nn.ConvTranspose2d(
                in_channels=self.num_filters,
                out_channels=observation_size[0],
                kernel_size=self.kernel_size,
                stride=stride,
                output_padding=padding,
            )
        )

    def forward(self, latent_obs):
        batch_size = latent_obs.size(0)

        # Fully connected layers with ReLu activations
        h_fc = torch.relu(self.linear_one(latent_obs))
        print(f"{h_fc.shape=}")
        h_fc = torch.relu(self.linear_two(h_fc))
        print(f"{h_fc.shape=}")
        h_fc = torch.relu(self.linear_three(h_fc))
        print(f"{h_fc.shape=}")

        deconv = h_fc.view(batch_size, self.num_filters, self.out_dim, self.out_dim)
        print(f"{deconv.shape=}")

        # Convolutional layers with ReLu activations
        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))
            print(f"{deconv.shape=}")

        # Sigmoid activation for final conv layer
        observation = torch.sigmoid(self.deconvs[-1](deconv))
        print(f"{observation.shape=}")

        return observation
