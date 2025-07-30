import torch
from torch import nn
from torch.nn import functional as F

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import DADSConfig


class DefaultSkillDynamicsModel(nn.Module):
    """
    DADS skill-conditioned forward model:
    p(s_{t+1} | s_t, z)
    Outputs mean and log-variance for next-state prediction.
    """

    def __init__(self, observation_size: int, num_skills: int):
        super().__init__()

        input_size = observation_size + num_skills

        self.network = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(),
        )

        self.mean_head = nn.Linear(256, observation_size)
        self.logvar_head = nn.Linear(256, observation_size)

        # reasonable clipping bounds for log-variance (std ≈ e^-5 to e^5)
        self.logvar_min = -5.0
        self.logvar_max = 5.0

    def forward(
        self, state: torch.Tensor, skill_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: (batch, state_dim)
        skill_onehot: (batch, skill_dim)
        """
        x = torch.cat([state, skill_onehot], dim=-1)
        features = self.network(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)  # <-- clamp here
        return mean, logvar


class SkillDynamicsModel(nn.Module):
    """
    DADS skill-conditioned forward model:
    p(s_{t+1} | s_t, z)
    Outputs mean and log-variance for next-state prediction.
    """

    def __init__(self, observation_size: int, num_skills: int, config: DADSConfig):
        super().__init__()

        input_size = observation_size + num_skills

        self.network = MLP(
            input_size=input_size,
            output_size=None,
            config=config.discriminator_config,
        )

        self.mean_head = nn.Linear(self.network.output_size, observation_size)
        self.logvar_head = nn.Linear(self.network.output_size, observation_size)

        # reasonable clipping bounds for log-variance (std ≈ e^-5 to e^5)
        self.logvar_min = -5.0
        self.logvar_max = 5.0

    def forward(
        self, state: torch.Tensor, skill_onehot: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        state: (batch, state_dim)
        skill_onehot: (batch, skill_dim)
        """
        x = torch.cat([state, skill_onehot], dim=-1)
        features = self.network(x)
        mean = self.mean_head(features)
        logvar = self.logvar_head(features)
        logvar = torch.clamp(logvar, self.logvar_min, self.logvar_max)  # <-- clamp here
        return mean, logvar
