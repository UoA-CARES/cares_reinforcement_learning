import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import TD7Config


class BaseActor(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions

        self.feature_layer: MLP | nn.Sequential = MLP(
            input_size=observation_size,
            output_size=config.zs_dim,
            config=config.feature_layer_config,
        )

        self.act_net: MLP | nn.Sequential = MLP(
            input_size=config.zs_dim + self.feature_layer.output_size,
            output_size=num_actions,
            config=config.actor_config,
        )

    def forward(self, state: torch.Tensor, zs: torch.Tensor) -> torch.Tensor:
        latent = self.feature_layer(state)
        latent = hlp.avg_l1_norm(latent)

        combined = torch.cat([latent, zs], dim=-1)

        output = self.act_net(combined)
        return output


class DefaultActor(BaseActor):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        # Skip BaseActor.__init__ and call nn.Module.__init__ directly
        # pylint: disable-next=non-parent-init-called
        nn.Module.__init__(self)

        # Set the attributes that BaseActor.forward() expects
        self.observation_size = observation_size
        self.num_actions = num_actions

        zs_dim = 256
        hidden_sizes = [256, 256]

        self.feature_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
        )

        self.act_net = nn.Sequential(
            nn.Linear(zs_dim + hidden_sizes[0], hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Tanh(),
        )


class Actor(BaseActor):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):

        super().__init__(
            observation_size=observation_size, num_actions=num_actions, config=config
        )
