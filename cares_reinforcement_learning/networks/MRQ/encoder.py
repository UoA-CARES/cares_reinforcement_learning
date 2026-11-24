"""MR.Q state/action representation and latent MDP prediction head."""

import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import MRQConfig
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseEncoder(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: MRQConfig):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.zs_dim = config.zs_dim
        self.zsa_dim = config.zsa_dim
        self.reward_bins = config.reward_bins

        self.state_encoder: MLP | nn.Sequential = MLP(
            input_size=observation_size,
            output_size=config.zs_dim,
            config=config.state_encoder_config,
        )
        self.action_encoder: MLP | nn.Sequential = MLP(
            input_size=num_actions,
            output_size=config.za_dim,
            config=config.action_encoder_config,
        )
        self.state_action_encoder: MLP | nn.Sequential = MLP(
            input_size=config.zs_dim + config.za_dim,
            output_size=config.zsa_dim,
            config=config.state_action_encoder_config,
        )
        # Intentionally linear: the MDP predictor operates directly on z_sa.
        self.model = nn.Linear(config.zsa_dim, 1 + config.zs_dim + config.reward_bins)

    def zs(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state)

    def zsa(self, zs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        za = self.action_encoder(action)
        return self.state_action_encoder(torch.cat([zs, za], dim=-1))

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zs = self.zs(state)
        return zs, self.zsa(zs, action)

    def model_all(
        self, zs: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        prediction = self.model(self.zsa(zs, action))
        return (
            prediction[..., :1],
            prediction[..., 1 : 1 + self.zs_dim],
            prediction[..., 1 + self.zs_dim :],
        )


class DefaultEncoder(BaseEncoder):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        nn.Module.__init__(self)  # pylint: disable=non-parent-init-called
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.zs_dim = 512
        self.zsa_dim = 512
        self.reward_bins = 65
        self.state_encoder = nn.Sequential(
            nn.Linear(observation_size, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ELU(),
        )
        self.action_encoder = nn.Sequential(nn.Linear(num_actions, 256), nn.ELU())
        self.state_action_encoder = nn.Sequential(
            nn.Linear(512 + 256, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512, elementwise_affine=False),
            nn.ELU(),
            nn.Linear(512, 512),
        )
        self.model = nn.Linear(512, 1 + 512 + 65)


class Encoder(BaseEncoder):
    def __init__(self, observation_size: int, num_actions: int, config: MRQConfig):
        super().__init__(observation_size, num_actions, config)
