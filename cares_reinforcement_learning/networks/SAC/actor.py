import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP, SquashedNormal
from cares_reinforcement_learning.util.configurations import SACConfig


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SACConfig):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_sizes = config.hidden_size_actor
        self.log_std_bounds = config.log_std_bounds

        # Default actor network should have this architecture with hidden_sizes = [256, 256]:
        # self.act_net = nn.Sequential(
        #     nn.Linear(observation_size, self.hidden_size[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size[0], self.hidden_size[1]),
        #     nn.ReLU(),
        # )

        self.act_net = MLP(
            self.observation_size,
            self.hidden_sizes,
            output_size=None,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        self.mean_linear = nn.Linear(self.hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(self.hidden_sizes[-1], num_actions)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act_net(state)
        mu = self.mean_linear(x)
        log_std = self.log_std_linear(x)

        # Bound the action to finite interval.
        # Apply an invertible squashing function: tanh
        # employ the change of variables formula to compute the likelihoods of the bounded actions

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)

        std = log_std.exp()

        dist = SquashedNormal(mu, std)
        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(-1, keepdim=True)

        return sample, log_pi, dist.mean
