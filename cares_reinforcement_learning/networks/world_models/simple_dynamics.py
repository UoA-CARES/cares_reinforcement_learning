import torch
import torch.nn.functional as F
import torch.utils
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp


class SimpleDynamics(nn.Module):
    """
    A world model with fully connected layers. It takes current states (s) and
    current actions (a), and predict next states (s').

    In this case, it predict the delta (s' - s) between s and s' to have a
    accurate estiamtion. Both input and label are normalized based on
    experience replay dataset.

    Normalization is recommanded. So it can be trained faster.

    :param (int) observation_size -- dimension of states
    :param (int) num_actions -- dimension of actions
    :param (int) hidden_size -- size of neurons in hidden layers.
    """

    def __init__(self, observation_size: int, num_actions: int, hidden_size: int):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.layer1 = nn.Linear(observation_size + num_actions, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.mean_layer = nn.Linear(hidden_size, observation_size)
        self.logvar_layer = nn.Linear(hidden_size, observation_size)

        self.apply(hlp.weight_init)

        self.statistics = {}

    def forward(
        self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward the inputs throught the network.

        :param (Tensors) obs -- dimension of states
        :param (Tensors) actions -- dimension of actions

        :return (Tensors) mean_deltas -- unnormalized delta between current
        and next
        :return (Tensors) normalized_mean -- normalized delta of mean for
        uncertainty estimation.
        :return (Tensors) normalized_var -- normalized delta of var for
        uncertainty estimation.
        """

        # Always normalized obs
        normalized_obs = hlp.normalize_observation(observation, self.statistics)

        x = torch.cat((normalized_obs, actions), dim=1)
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)

        normalized_mean = self.mean_layer(x)
        logvar = self.logvar_layer(x)
        logvar = torch.tanh(logvar)
        normalized_var = torch.exp(logvar)

        # Always denormalized delta
        mean_deltas = hlp.denormalize_observation_delta(
            normalized_mean, self.statistics
        )
        return mean_deltas, normalized_mean, normalized_var
