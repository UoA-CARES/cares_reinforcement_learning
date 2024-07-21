import torch
from torch import nn

from cares_reinforcement_learning.util.common import SquashedNormal


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] = None,
        log_std_bounds: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [256, 256]
        if log_std_bounds is None:
            log_std_bounds = [-20, 2]

        self.hidden_size = hidden_size
        self.log_std_bounds = log_std_bounds

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
        )

        self.action_linear = nn.Linear(self.hidden_size[1], num_actions)

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.act_net(state)

        logits = self.action_linear(x)
        action_probs = torch.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(action_probs)
        sample = dist.sample()
        log_pi = dist.log_prob(sample)
        maxp_action = dist.probs.argmax() / len(dist.probs[0])

        return sample, log_pi, maxp_action
