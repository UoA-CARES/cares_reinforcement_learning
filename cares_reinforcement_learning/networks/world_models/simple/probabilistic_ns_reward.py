import torch
from torch import nn, Tensor
import torch.nn.functional as F
from cares_reinforcement_learning.util import weight_init_pnn, MLP, weight_init


class Probabilistic_NS_Reward(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, hidden_size: list, normalize:bool):
        """
        Note, This reward function is limited to 0 ~ 1 for dm_control.
        A reward model with fully connected layers. It takes current states (s)
        and current actions (a), and predict rewards (r).
        """
        super().__init__()
        print("Create a Prob NS Rewrad")
        self.normalize = normalize
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.model = MLP(input_size=observation_size, hidden_sizes=hidden_size, output_size=2)
        self.add_module('mlp', self.model)
        self.model.apply(weight_init)
    def forward(
            self,
            next_observation: torch.Tensor) -> tuple[Tensor, Tensor]:
        """
        Forward the inputs throught the network.
        Note: For DMCS environment, the reward is from 0~1.
        """
        pred = self.model(next_observation)
        var_mean = pred[:, 1].unsqueeze(dim=1)
        rwd_mean = pred[:, 0].unsqueeze(dim=1)
        logvar = torch.tanh(var_mean)
        normalized_var = torch.exp(logvar)
        if self.normalize:
            rwd_mean = F.sigmoid(rwd_mean)
        return rwd_mean, normalized_var
