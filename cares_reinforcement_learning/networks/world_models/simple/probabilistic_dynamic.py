import torch
import torch.utils
from torch import nn

from cares_reinforcement_learning.util import weight_init_pnn, MLP


class Probabilistic_Dynamics(nn.Module):
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

    def __init__(self, observation_size: int, num_actions: int, hidden_size: list):
        print("Create a Prob Dynamics")
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.model = MLP(input_size=observation_size + num_actions,
                         hidden_sizes=hidden_size,
                         output_size=2 * observation_size)

        self.add_module('mlp', self.model)

        self.model.apply(weight_init_pnn)

        self.statistics = {}

    def forward(
            self, observation: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
        assert (
                observation.shape[1] + actions.shape[1]
                == self.observation_size + self.num_actions
        )
        # Always normalized obs
        x = torch.cat((observation, actions), dim=1)
        pred = self.model(x)
        logvar = pred[:, :self.observation_size]
        normalized_mean = pred[:, self.observation_size:]
        logvar = torch.tanh(logvar)
        normalized_var = torch.exp(logvar)
        # Always denormalized delta
        return normalized_mean, normalized_var
