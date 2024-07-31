"""

Original Code: https://github.com/YannDubs/disentangling-vae/tree/master
"""

import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.encoders.autoencoder import Autoencoder
from cares_reinforcement_learning.networks.encoders.constants import Autoencoders
from cares_reinforcement_learning.networks.encoders.losses import BaseBurgessLoss


class BurgessAutoencoder(Autoencoder):
    def __init__(
        self,
        observation_size: tuple[int],
        latent_dim: int,
        loss_function: BaseBurgessLoss,
        num_layers: int = 4,
        num_filters: int = 32,
        kernel_size: int = 3,
        encoder_optimiser_params: dict[str, any] = None,
        decoder_optimiser_params: dict[str, any] = None,
    ):
        if encoder_optimiser_params is None:
            encoder_optimiser_params = {"lr": 1e-3}
        if decoder_optimiser_params is None:
            decoder_optimiser_params = {"lr": 1e-3, "weight_decay": 1e-7}

        super().__init__(
            ae_type=Autoencoders.BURGESS,
            loss_function=loss_function,
            observation_size=observation_size,
            latent_dim=latent_dim,
        )

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

        # Autoencoder Optimizer
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            **encoder_optimiser_params,
        )

        self.decoder_optimizer = torch.optim.Adam(
            self.decoder.parameters(),
            **decoder_optimiser_params,
        )

        # self.apply(weight_init_burgess)

    def update_autoencoder(self, data: torch.Tensor):
        # TODO handle
        output = self.forward(data, is_train=True)
        ae_loss = output["loss"]

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        ae_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

    def forward(
        self,
        observation,
        detach_cnn: bool = False,
        detach_output: bool = False,
        is_train=False,
        **kwargs,
    ):
        latent_dist = self.encoder(
            observation, detach_cnn=detach_cnn, detach_output=detach_output
        )

        mu, logvar, latent_sample = latent_dist

        reconstructed_observation = self.decoder(latent_sample)

        loss = self.loss_function(
            data=observation,
            reconstructed_data=reconstructed_observation,
            latent_dist=latent_dist,
            is_train=is_train,
            latent_sample=latent_sample,
            autoencoder=self,
        )

        if detach_output:
            latent_sample = latent_sample.detach()

        return {
            "latent_observation": latent_sample,
            "reconstructed_observation": reconstructed_observation,
            "latent_distribution": {"mu": mu, "logvar": logvar},
            "loss": loss,
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

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))

        return conv

    def _reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + std * eps

    def _enc_forward(
        self, obs: torch.Tensor, detach_cnn: bool = False, detach_output: bool = False
    ):
        batch_size = obs.size(0)

        # Convolutional layers with ReLu activations
        h = self._forward_conv(obs)

        # SAC AE detaches at the CNN layer
        if detach_cnn:
            h = h.detach()

        # Fully connected layers with ReLu activations
        h = h.view((batch_size, -1))
        h = torch.relu(self.linear_one(h))
        h = torch.relu(self.linear_two(h))

        # Fully connected layer for log variance and mean
        # Log std-dev in paper (bear in mind)
        mu_logvar = self.mu_logvar_gen(h)

        mu, logvar = mu_logvar.view(-1, self.latent_dim, 2).unbind(-1)

        if detach_output:
            mu = mu.detach()
            logvar = logvar.detach()

        latent_sample = self._reparameterize(mu, logvar)

        return mu, logvar, latent_sample

    def forward(self, obs, detach_cnn: bool = False, detach_output: bool = False):
        mu, logvar, latent_sample = self._enc_forward(
            obs, detach_cnn=detach_cnn, detach_output=detach_output
        )
        return mu, logvar, latent_sample


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
        h_fc = torch.relu(self.linear_two(h_fc))
        h_fc = torch.relu(self.linear_three(h_fc))

        deconv = h_fc.view(batch_size, self.num_filters, self.out_dim, self.out_dim)

        # Convolutional layers with ReLu activations
        for i in range(0, self.num_layers - 1):
            deconv = torch.relu(self.deconvs[i](deconv))

        # Sigmoid activation for final conv layer
        observation = torch.sigmoid(self.deconvs[-1](deconv))

        return observation
