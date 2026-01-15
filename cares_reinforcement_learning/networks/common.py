from __future__ import annotations

from typing import Any

import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions import Normal
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.vanilla_autoencoder import (
    Encoder,
    VanillaAutoencoder,
)
from cares_reinforcement_learning.util.network_configurations import MLPConfig
from cares_reinforcement_learning.networks.mlp import MLP


class BasePolicy(nn.Module):
    def __init__(self, input_size: int, num_actions: int, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions

    def forward(self, state: torch.Tensor) -> Any:
        raise NotImplementedError("Subclasses should implement this method.")


class DeterministicPolicy(BasePolicy):
    def __init__(self, input_size: int, num_actions: int, config: MLPConfig):
        super().__init__(input_size, num_actions)

        self.act_net: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=self.num_actions,
            config=config,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output


class GaussianPolicy(BasePolicy):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        log_std_bounds: list[float],
        config: MLPConfig,
    ):
        super().__init__(input_size, num_actions)

        self.log_std_bounds = log_std_bounds

        self.act_net: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=None,
            config=config,
        )

        self.mean_linear = nn.Linear(self.act_net.output_size, num_actions)
        self.log_std_linear = nn.Linear(self.act_net.output_size, num_actions)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act_net(state)
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std_min, log_std_max = self.log_std_bounds
        log_std = torch.clamp(log_std, log_std_min, log_std_max)

        std = log_std.exp()

        dist = Normal(mu, std)
        sample = dist.rsample()  # Sample from the Gaussian distribution
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean


class TanhGaussianPolicy(BasePolicy):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        log_std_bounds: list[float],
        config: MLPConfig,
    ):
        super().__init__(input_size, num_actions)

        self.log_std_bounds = log_std_bounds

        self.act_net: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=None,
            config=config,
        )

        self.mean_linear = nn.Linear(self.act_net.output_size, num_actions)
        self.log_std_linear = nn.Linear(self.act_net.output_size, num_actions)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act_net(state)
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Bound the action to finite interval.
        # Apply an invertible squashing function: tanh
        # employ the change of variables formula to compute the likelihoods of the bounded actions

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean


class BaseCritic(nn.Module):
    def __init__(self, input_size: int, output_size: int, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses should implement this method.")


class QNetwork(BaseCritic):
    def __init__(self, input_size: int, output_size: int, config: MLPConfig):
        super().__init__(input_size, output_size)

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            config=config,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat([state, action], dim=1)
        q = self.Q(obs_action)
        return q


class TwinQNetwork(BaseCritic):
    def __init__(self, input_size: int, output_size: int, config: MLPConfig):
        super().__init__(input_size, output_size)

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            config=config,
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            config=config,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2


class ContinuousDistributedCritic(BaseCritic):
    def __init__(self, input_size: int, output_size: int, config: MLPConfig):
        super().__init__(input_size, output_size)

        self.mean_layer: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            config=config,
        )

        self.std_layer: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=self.output_size,
            config=config,
        )

        self.soft_std_layer = nn.Softplus()

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        mean = self.mean_layer(obs_action)
        # Add a small value to the standard deviation to prevent division by zero
        std = self.std_layer(obs_action)
        std = self.soft_std_layer(std) + 1e-6
        return mean, std


class EnsembleCritic(BaseCritic):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        ensemble_size: int,
        config: MLPConfig,
        critic_type: type[BaseCritic],
    ):
        super().__init__(input_size, output_size)
        self.ensemble_size = ensemble_size

        self.critics: list[BaseCritic | nn.Sequential] = []

        for i in range(self.ensemble_size):
            critic_net = critic_type(
                input_size=self.input_size, output_size=self.output_size, config=config
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.critics.append(critic_net)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        values = torch.stack(
            tuple(critic(state, action) for critic in self.critics), dim=1
        )
        return values


# TODO generalise detach - cnn or output
class EncoderPolicy(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        actor: BasePolicy,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.num_actions = actor.num_actions
        self.encoder = encoder
        self.actor = actor

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(  # type: ignore
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        actor_input = state_latent
        if self.add_vector_observation:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return self.actor(actor_input)


class EncoderCritic(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        critic: BaseCritic,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.critic = critic

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        detach_encoder: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        critic_input = state_latent
        if self.add_vector_observation:
            critic_input = torch.cat([state["vector"], critic_input], dim=1)

        return self.critic(critic_input, action)


class AEActor(nn.Module):
    def __init__(
        self,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        actor: BasePolicy,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.num_actions = actor.num_actions
        self.autoencoder = autoencoder
        self.actor = actor

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # NaSATD3 detatches the encoder at the output
        if self.autoencoder.ae_type == Autoencoders.BURGESS:
            # take the mean value for stability
            z_vector, _, _ = self.autoencoder.encoder(
                state["image"], detach_output=detach_encoder
            )
        else:
            z_vector = self.autoencoder.encoder(
                state["image"], detach_output=detach_encoder
            )

        actor_input = z_vector
        if self.add_vector_observation:
            actor_input = torch.cat([state["vector"], actor_input], dim=1)

        return self.actor(actor_input)


class AECritc(nn.Module):
    def __init__(
        self,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        critic: BaseCritic,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.autoencoder = autoencoder
        self.critic = critic

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(
        self,
        state: dict[str, torch.Tensor],
        action: torch.Tensor,
        detach_encoder: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NaSATD3 detatches the encoder at the output
        if self.autoencoder.ae_type == Autoencoders.BURGESS:
            # take the mean value for stability
            z_vector, _, _ = self.autoencoder.encoder(
                state["image"], detach_output=detach_encoder
            )
        else:
            z_vector = self.autoencoder.encoder(
                state["image"], detach_output=detach_encoder
            )

        critic_input = z_vector
        if self.add_vector_observation:
            critic_input = torch.cat([state["vector"], critic_input], dim=1)

        return self.critic(critic_input, action)


# Stable version of the Tanh transform - overriden to avoid NaN values through atanh in pytorch
class StableTanhTransform(TanhTransform):
    def __init__(self, cache_size=1):
        super().__init__(cache_size=cache_size)

    @staticmethod
    def atanh(x):
        return 0.5 * (x.log1p() - (-x).log1p())

    def __eq__(self, other):
        return isinstance(other, StableTanhTransform)

    def _inverse(self, y):
        # We do not clamp to the boundary here as it may degrade the performance of certain algorithms.
        # one should use `cache_size=1` instead
        return self.atanh(y)


# These methods are not required for the purposes of SAC and are thus intentionally ignored
# pylint: disable=abstract-method
class SquashedNormal(TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale
        self.base_dist = pyd.Normal(loc, scale)

        transforms = [StableTanhTransform()]
        super().__init__(self.base_dist, transforms, validate_args=False)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
