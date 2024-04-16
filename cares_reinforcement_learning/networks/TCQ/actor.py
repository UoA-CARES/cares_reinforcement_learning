import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch import distributions as pyd

from cares_reinforcement_learning.networks.TCQ import Mlp


# These methods are not required for the purposes of SAC and are thus intentionally ignored
# pylint: disable=abstract-method
class TQCTanhNormal(pyd.Distribution):
    def __init__(self, normal_mean, normal_std):
        super().__init__()
        self.normal_mean = normal_mean
        self.normal_std = normal_std

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.standard_normal = pyd.Normal(
            torch.zeros_like(self.normal_mean, device=device),
            torch.ones_like(self.normal_std, device=device),
        )
        self.normal = pyd.Normal(normal_mean, normal_std)

    def log_prob(self, pre_tanh):
        log_det = (
            2 * np.log(2) + F.logsigmoid(2 * pre_tanh) + F.logsigmoid(-2 * pre_tanh)
        )
        result = self.normal.log_prob(pre_tanh) - log_det
        return result

    def rsample(self):
        pretanh = self.normal_mean + self.normal_std * self.standard_normal.sample()
        return torch.tanh(pretanh), pretanh


class Actor(nn.Module):
    def __init__(self, observation_size, num_actions):
        super().__init__()
        self.action_dim = num_actions
        self.net = Mlp(observation_size, [256, 256], 2 * num_actions)

    def forward(self, state):
        mean, log_std = self.net(state).split([self.action_dim, self.action_dim], dim=1)
        log_std = log_std.clamp(-20, 2)

        if self.training:
            std = torch.exp(log_std)
            tanh_normal = TQCTanhNormal(mean, std)
            action, pre_tanh = tanh_normal.rsample()
            log_prob = tanh_normal.log_prob(pre_tanh)
            log_prob = log_prob.sum(dim=1, keepdim=True)
        else:  # deterministic eval without log_prob computation
            action = torch.tanh(mean)
            log_prob = None
        return action, log_prob

    def select_action(self, state):
        action, _ = self.forward(state)
        action = action[0].cpu().detach().numpy()
        return action
