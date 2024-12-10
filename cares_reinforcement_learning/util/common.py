from typing import Callable

import torch
from torch import distributions as pyd
from torch import nn
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.transforms import TanhTransform

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.encoders.burgess_autoencoder import BurgessAutoencoder
from cares_reinforcement_learning.encoders.constants import Autoencoders
from cares_reinforcement_learning.encoders.vanilla_autoencoder import (
    Encoder,
    VanillaAutoencoder,
)
from cares_reinforcement_learning.util.configurations import MLPConfig


def get_pytorch_module_from_name(module_name: str) -> Callable[..., nn.Module] | None:
    if module_name == "":
        return None
    return getattr(nn, module_name)


# Standard Multilayer Perceptron (MLP) network
class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int | None,
        config: MLPConfig,
    ):
        super().__init__()

        hidden_sizes = config.hidden_sizes

        layer_order = config.layer_order

        batch_layer = get_pytorch_module_from_name(config.batch_layer)

        dropout_layer = get_pytorch_module_from_name(config.dropout_layer)

        norm_layer = get_pytorch_module_from_name(config.norm_layer)

        hidden_activation_function = get_pytorch_module_from_name(
            config.hidden_activation_function
        )

        output_activation_function = get_pytorch_module_from_name(
            config.output_activation_function
        )

        layers = nn.ModuleList()

        for next_size in hidden_sizes:
            layers.append(nn.Linear(input_size, next_size))

            for layer_type in layer_order:

                if (
                    layer_type == "activation"
                    and hidden_activation_function is not None
                ):
                    layers.append(
                        hidden_activation_function(
                            **config.hidden_activation_function_args
                        )
                    )

                elif layer_type == "batch" and batch_layer is not None:
                    layers.append(batch_layer(next_size, **config.batch_layer_args))

                elif layer_type == "layernorm" and norm_layer is not None:
                    layers.append(norm_layer(next_size, **config.norm_layer_args))

                elif layer_type == "dropout" and dropout_layer is not None:
                    layers.append(dropout_layer(**config.dropout_layer_args))

            input_size = next_size

        if output_size is not None:
            layers.append(nn.Linear(input_size, output_size))

            if output_activation_function is not None:
                layers.append(
                    output_activation_function(**config.output_activation_function_args)
                )

        self.model = nn.Sequential(*layers)

    def forward(self, state):
        return self.model(state)


class DeterministicPolicy(nn.Module):
    def __init__(self, input_size: int, num_actions: int, config: MLPConfig):
        super().__init__()

        self.observation_size = input_size
        self.num_actions = num_actions

        self.act_net: MLP | nn.Sequential = MLP(
            input_size=self.observation_size,
            output_size=self.num_actions,
            config=config,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output


class TanhGaussianPolicy(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_actions: int,
        log_std_bounds: list[float],
        config: MLPConfig,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_actions = num_actions
        self.log_std_bounds = log_std_bounds

        self.act_net: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=None,
            config=config,
        )

        self.mean_linear = nn.Linear(config.hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(config.hidden_sizes[-1], num_actions)

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


class QValueCritic(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: MLPConfig):
        super().__init__()

        self.input_size = input_size

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=output_size,
            config=config,
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        obs_action = torch.cat([state, action], dim=1)
        q = self.Q(obs_action)
        return q


class TwinQValueCritic(nn.Module):
    def __init__(self, input_size: int, output_size: int, config: MLPConfig):
        super().__init__()

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=output_size,
            config=config,
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=output_size,
            config=config,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2


class ContinuousDistributedCritic(nn.Module):
    def __init__(
        self, input_size: int, mean_layer_config: MLPConfig, std_layer_config: MLPConfig
    ):
        super().__init__()

        self.mean_layer: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=1,
            config=mean_layer_config,
        )

        self.std_layer: MLP | nn.Sequential = MLP(
            input_size=input_size,
            output_size=1,
            config=std_layer_config,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        mean = self.mean_layer(obs_action)
        # Add a small value to the standard deviation to prevent division by zero
        std = self.std_layer(obs_action) + 1e-6
        return mean, std


class EncoderPolicy(nn.Module):
    def __init__(
        self,
        num_actions: int,
        encoder: Encoder,
        actor: DeterministicPolicy | TanhGaussianPolicy,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.encoder = encoder
        self.actor = actor

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(  # type: ignore
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        critic: QValueCritic | TwinQValueCritic,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Detach at the CNN layer to prevent backpropagation through the encoder
        state_latent = self.encoder(state["image"], detach_cnn=detach_encoder)

        critic_input = state_latent
        if self.add_vector_observation:
            critic_input = torch.cat([state["vector"], critic_input], dim=1)

        return self.critic(critic_input, action)


class AEActor(nn.Module):
    def __init__(
        self,
        num_actions: int,
        autoencoder: VanillaAutoencoder | BurgessAutoencoder,
        actor: DeterministicPolicy | TanhGaussianPolicy,
        add_vector_observation: bool = False,
    ):
        super().__init__()

        self.num_actions = num_actions
        self.autoencoder = autoencoder
        self.actor = actor

        self.add_vector_observation = add_vector_observation

        self.apply(hlp.weight_init)

    def forward(
        self, state: dict[str, torch.Tensor], detach_encoder: bool = False
    ) -> torch.Tensor:
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
        critic: QValueCritic | TwinQValueCritic,
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
