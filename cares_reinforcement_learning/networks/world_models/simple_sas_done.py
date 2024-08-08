import torch
from torch import nn
import torch.nn.functional as F
from cares_reinforcement_learning.util.helpers import weight_init


class Simple_SAS_Done(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, hidden_size: int):
        """
        Note, This reward function is limited to 0 ~ 1 for dm_control.
        A reward model with fully connected layers. It takes current states (s)
        and current actions (a), and predict rewards (r).

        :param (int) observation_size -- dimension of states
        :param (int) num_actions -- dimension of actions
        :param (int) hidden_size -- size of neurons in hidden layers.
        """
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.linear1 = nn.Linear(2 * observation_size + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.apply(weight_init)

    def forward(
        self, observation: torch.Tensor, actions: torch.Tensor, next_observation: torch.Tensor, normalized: bool = False
    ) -> torch.Tensor:
        """
        Forward the inputs throught the network.
        Note: For DMCS environment, the reward is from 0~1.

        :param (Tensors) obs -- dimension of states
        :param (Tensors) actions -- dimension of actions
        :param (Bool) normalized -- whether normalized reward to 0~1

        :return (Tensors) x -- predicted rewards.
        """
        assert (
            observation.shape[1] + actions.shape[1]
            == self.observation_size + self.num_actions
        )
        x = torch.cat((observation, actions, next_observation), dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        prob_x = F.sigmoid(x)
        return prob_x