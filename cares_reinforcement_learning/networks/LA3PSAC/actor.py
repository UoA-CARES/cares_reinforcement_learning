import torch
from torch import nn
from torch.nn import functional as F

from cares_reinforcement_learning.util.common import SquashedNormal
from torch.distributions import Normal


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, num_inputs: int, num_actions: int, action_space=None):
        super(Actor).__init__()
        self.hidden_dim = [256, 256]
        self.log_std_bounds = [-20, 2]
        self.epsilon = 1e-6
        
        self.linear1 = nn.Linear(num_inputs, self.hidden_dim[0])
        self.linear2 = nn.Linear(self.hidden_dim[0], self.hidden_dim[1])

        self.mean_linear = nn.Linear(self.hidden_dim[1], num_actions)
        self.log_std_linear = nn.Linear(self.hidden_dim[1], num_actions)

        self.apply(self.weights_init_)

        # Action rescaling
        if action_space is None:
            self.action_scale = torch.tensor(1.)
            self.action_bias = torch.tensor(0.)
        else:
            self.action_scale = torch.FloatTensor(
                (action_space.high - action_space.low) / 2.)
            self.action_bias = torch.FloatTensor(
                (action_space.high + action_space.low) / 2.)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))

        mean = self.mean_linear(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=self.log_std_bounds[0], max=self.log_std_bounds[1])

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(mean, std)

        x_t = normal.rsample()
        y_t = torch.tanh(x_t)

        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)

        # Enforcing action bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + self.epsilon)
        log_prob = log_prob.sum(1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale + self.action_bias

        return action, log_prob, mean

    def weights_init_(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight, gain=1)
            torch.nn.init.constant_(m.bias, 0)

        
