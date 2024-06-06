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
        states = states / 255

        latent_dist = self.encoder(states, detach)

        latent_sample = self.reparameterize(*latent_dist)

        rec_obs = self.decoder(latent_sample)
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
            "latent_dist": latent_dist,
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

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(
                    observation_size[0],
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=1,
                )
            ]
        )

        self.out_dim = hlp.flatten(observation_size[1], k=self.kernel_size, s=2, p=1)

        for _ in range(self.num_layers - 1):
            self.convs.append(
                nn.Conv2d(
                    self.num_filters,
                    self.num_filters,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=1,
                )
            )
            self.out_dim = hlp.flatten(self.out_dim, k=self.kernel_size, s=2, p=1)

        self.n_flatten = self.out_dim * self.out_dim * self.num_filters

        # Fully connected layers
        self.linear_one = nn.Linear(self.n_flatten, hidden_dim)
        self.linear_two = nn.Linear(hidden_dim, hidden_dim)

        # Fully connected layers for mean and variance
        self.mu_logvar_gen = nn.Linear(hidden_dim, self.latent_dim * 2)

    def _forward_conv(self, x: torch.Tensor) -> torch.Tensor:
        conv = torch.relu(self.convs[0](x))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        return conv

    def enc_forward(self, x, detach=False):
        batch_size = x.size(0)

        # Convolutional layers with ReLu activations
        x = self._forward_conv(x)
        print(f"{x.shape=}")

        # Fully connected layers with ReLu activations
        x = x.view((batch_size, -1))
        x = torch.relu(self.linear_one(x))
        x = torch.relu(self.linear_two(x))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(x)
        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        if detach:
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
        kernel_size: int = 3,
    ):
        super().__init__()

        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.img_size = observation_size

        # Shape required to start transpose convs
        self.reshape = (hid_channels, kernel_size, kernel_size)
        n_chan = self.img_size[0]
        self.img_size = observation_size

        # Fully connected layers
        self.lin1 = nn.Linear(latent_dim, hidden_dim)
        self.lin2 = nn.Linear(hidden_dim, hidden_dim)
        self.lin3 = nn.Linear(hidden_dim, np.prod(self.reshape))

        # Convolutional layers
        cnn_kwargs = dict(stride=2, padding=1)
        # If input image is 64x64 do fourth convolution
        if self.img_size[1] == self.img_size[2] == 64:
            self.convT_64 = nn.ConvTranspose2d(
                hid_channels, hid_channels, kernel_size, **cnn_kwargs
            )

        self.convT1 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs
        )
        self.convT2 = nn.ConvTranspose2d(
            hid_channels, hid_channels, kernel_size, **cnn_kwargs
        )
        self.convT3 = nn.ConvTranspose2d(
            hid_channels, n_chan, kernel_size, **cnn_kwargs
        )

    def forward(self, z):
        batch_size = z.size(0)

        # Fully connected layers with ReLu activations
        x = torch.relu(self.lin1(z))
        x = torch.relu(self.lin2(x))
        x = torch.relu(self.lin3(x))
        x = x.view(batch_size, *self.reshape)

        # Convolutional layers with ReLu activations
        if self.img_size[1] == self.img_size[2] == 64:
            x = torch.relu(self.convT_64(x))
        x = torch.relu(self.convT1(x))
        x = torch.relu(self.convT2(x))
        # Sigmoid activation for final conv layer
        x = torch.sigmoid(self.convT3(x))

        return x


# TODO Remove here before final pull request


def main():
    ae = BurgessAutoencoder((3, 64, 64), 10, num_filters=16, kernel_size=4)

    x = torch.randn(1, 3, 64, 64)
    out = ae(x)


if __name__ == "__main__":
    main()
