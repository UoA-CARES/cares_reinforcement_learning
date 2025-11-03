"""
Ensemble of Predictive Discrete Model (EPPM)
Predict outputs  a point estimate e.g. discrete value
"""

import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import NaSATD3Config


class BaseEPDM(nn.Module):
    def __init__(self, prediction_net: nn.Module):
        super().__init__()

        self.prediction_net = prediction_net

        self.apply(hlp.weight_init)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        x = torch.cat([state, action], dim=1)
        out = self.prediction_net(x)
        return out


class DefaultEPDM(BaseEPDM):
    def __init__(self, observation_size: int, num_actions: int):
        input_dim = observation_size + num_actions
        output_dim = observation_size

        hidden_sizes = [512, 512]

        prediction_net = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[0], out_features=hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(in_features=hidden_sizes[1], out_features=output_dim),
        )

        super().__init__(prediction_net=prediction_net)


# TODO make this fully MLP
# pylint: disable-next=invalid-name
class EPDM(BaseEPDM):
    def __init__(self, observation_size: int, num_actions: int, config: NaSATD3Config):
        input_size = observation_size + num_actions
        output_dim = observation_size

        prediction_net = MLP(
            input_size=input_size,
            output_size=output_dim,
            config=config.epm_config,
        )

        super().__init__(prediction_net=prediction_net)
