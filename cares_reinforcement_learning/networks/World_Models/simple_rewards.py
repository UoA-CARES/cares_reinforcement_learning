import torch
from torch import nn
import torch.nn.functional as F
from cares_reinforcement_learning.util.helpers import weight_init


class Simple_Reward(nn.Module):
    def __init__(self, observation_size, num_actions, hidden_size):
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
        self.linear1 = nn.Linear(observation_size+num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)
        self.apply(weight_init)

    def forward(self, obs, actions):
        """
        Forward the inputs throught the network.

        :param (Tensors) obs -- dimension of states
        :param (Tensors) actions -- dimension of actions

        :return (Tensors) x -- predicted rewards.
        """
        assert (obs.shape[1]+actions.shape[1] == self.observation_size +
                self.num_actions)
        x = torch.cat((obs, actions), dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)
        x = self.linear3(x)
        x = F.sigmoid(x)
        return x
