import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import BROConfig
from cares_reinforcement_learning.networks.bro_architecture import bro_weight_init
from cares_reinforcement_learning.networks.distributions import SquashedNormal
from cares_reinforcement_learning.networks.mlp_architecture import MLP


class OptimisticActor(nn.Module):
    """
    BRO optimistic exploration policy.

    Predicts a shift to the exploitation actor's Gaussian mean.

    The standard deviation is inherited from the exploitation actor
    and multiplied by std_multiplier.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: BROConfig,
    ):
        super().__init__()

        self.act_net = MLP(
            input_size=observation_size + num_actions,
            output_size=None,
            config=config.optimistic_actor_config,
        )

        self.mean_shift = nn.Linear(
            self.act_net.output_size,
            num_actions,
            bias=False,
        )

        self.apply(bro_weight_init)

        # Small initial adjustment, as in the original implementation.
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

        return (action, log_pi, dist.mean, optimistic_means, optimistic_stds)
