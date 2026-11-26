import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import BROConfig
from cares_reinforcement_learning.networks.bro_architecture import (
    BroNetBlock,
    bro_weight_init,
)
from cares_reinforcement_learning.networks.distributions import SquashedNormal
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class BaseOptimisticActor(nn.Module):
    """BRO optimistic exploration policy."""

    def __init__(self, act_net: nn.Module, mean_shift: nn.Linear):
        super().__init__()

        self.act_net = act_net
        self.mean_shift = mean_shift

        self.apply(bro_weight_init)

        # Small initial mean adjustment, matching the original BRO.
        nn.init.orthogonal_(self.mean_shift.weight, gain=0.01)

    def gaussian_parameters(
        self,
        state: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        std_multiplier: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Construct the optimistic Gaussian parameters."""
        inputs = torch.cat([state, means], dim=-1)

        mean_shift = self.mean_shift(self.act_net(inputs))

        optimistic_means = means + mean_shift
        optimistic_stds = stds * std_multiplier

        return optimistic_means, optimistic_stds

    def forward(
        self,
        state: torch.Tensor,
        means: torch.Tensor,
        stds: torch.Tensor,
        std_multiplier: float = 0.75,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        optimistic_means, optimistic_stds = self.gaussian_parameters(
            state, means, stds, std_multiplier
        )

        dist = SquashedNormal(optimistic_means, optimistic_stds)

        action = dist.rsample()
        log_pi = dist.log_prob(action).sum(-1, keepdim=True)

        return action, log_pi, dist.mean, optimistic_means, optimistic_stds


class DefaultOptimisticActor(BaseOptimisticActor):
    """Default BRO optimistic actor: width 256, one BroNet block."""

    def __init__(self, observation_size: int, num_actions: int):
        act_net = nn.Sequential(
            nn.Linear(observation_size + num_actions, 256),
            nn.LayerNorm(256, eps=1e-6),
            nn.ReLU(),
            BroNetBlock(256),
        )

        mean_shift = nn.Linear(256, num_actions, bias=False)

        super().__init__(act_net, mean_shift)


class OptimisticActor(BaseOptimisticActor):
    """Configurable BRO optimistic exploration actor."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: BROConfig,
    ):
        act_net = MLP(
            input_size=observation_size + num_actions,
            output_size=None,
            config=config.optimistic_actor_config,
        )

        mean_shift = nn.Linear(
            act_net.output_size,
            num_actions,
            bias=False,
        )

        super().__init__(act_net, mean_shift)
