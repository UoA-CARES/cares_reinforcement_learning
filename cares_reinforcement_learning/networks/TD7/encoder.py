import torch
import torch.nn as nn
import torch.nn.functional as F

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import TD7Config


class BaseEncoder(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions

        self.activ = F.elu
        self.zs_dim = config.zs_dim

        # state encoder
        self.state_encoder: MLP | nn.Sequential = MLP(
            input_size=observation_size,
            output_size=self.zs_dim,
            config=config.state_encoder_config,
        )

        # state-action encoder
        self.state_action_encoder: MLP | nn.Sequential = MLP(
            input_size=self.zs_dim + self.num_actions,
            output_size=self.zs_dim,
            config=config.state_action_encoder_config,
        )

    def zs(self, state):
        zs = self.state_encoder(state)
        zs = hlp.avg_l1_norm(zs)
        return zs

    def zsa(self, zs, action):
        zsa = torch.cat([zs, action], 1)
        zsa = self.state_action_encoder(zsa)
        return zsa

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zs = self.zs(state)
        zsa = self.zsa(zs, action)
        return zs, zsa


class DefaultEncoder(BaseEncoder):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        # Skip BaseEncoder.__init__ and call nn.Module.__init__ directly
        # pylint: disable-next=non-parent-init-called
        nn.Module.__init__(self)

        # Set the attributes that BaseEncoder.forward() expects
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.zs_dim = 256

        hdim = 256

        self.activ = F.elu

        # state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(self.observation_size, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
            nn.Linear(hdim, self.zs_dim),
        )

        # state-action encoder
        self.state_action_encoder = nn.Sequential(
            nn.Linear(self.zs_dim + self.num_actions, hdim),
            nn.ELU(),
            nn.Linear(hdim, hdim),
            nn.ELU(),
            nn.Linear(hdim, self.zs_dim),
        )


class Encoder(BaseEncoder):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )
