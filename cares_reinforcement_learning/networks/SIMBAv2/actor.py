import torch
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import SIMBAV2Config
from cares_reinforcement_learning.networks.common import BasePolicy
from cares_reinforcement_learning.networks.distributions import SquashedNormal
from cares_reinforcement_learning.networks.simba_architecture import (
    SIMBAV2Encoder,
    SIMBAV2Linear,
    SIMBAV2Scaler,
)


class DefaultActor(BasePolicy):
    """
    Default SimbaV2 actor architecture.

    Actor:
        hyperspherical input embedding
        -> 1 SimbaV2 block, hidden size 128
        -> separate mean and log-std prediction heads
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: int = 128,
        num_blocks: int = 1,
        c_shift: float = 3.0,
        log_std_bounds: list[float] | None = None,
    ):
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
        )

        if log_std_bounds is None:
            log_std_bounds = [-10.0, 2.0]

        self.log_std_bounds = log_std_bounds

        self.encoder = SIMBAV2Encoder(
            input_size=observation_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            c_shift=c_shift,
        )

        # Mean prediction head
        self.mean_linear1 = SIMBAV2Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.mean_scaler = SIMBAV2Scaler(size=hidden_size, init=1.0, scale=1.0)
        self.mean_linear2 = SIMBAV2Linear(
            in_features=hidden_size, out_features=num_actions
        )
        self.mean_bias = nn.Parameter(torch.zeros(num_actions))

        # Log standard-deviation prediction head
        self.log_std_linear1 = SIMBAV2Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.log_std_scaler = SIMBAV2Scaler(size=hidden_size, init=1.0, scale=1.0)
        self.log_std_linear2 = SIMBAV2Linear(
            in_features=hidden_size, out_features=num_actions
        )
        self.log_std_bias = nn.Parameter(torch.zeros(num_actions))

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(state)

        mean = self.mean_linear1(features)
        mean = self.mean_scaler(mean)
        mean = self.mean_linear2(mean) + self.mean_bias

        log_std = self.log_std_linear1(features)
        log_std = self.log_std_scaler(log_std)
        log_std = self.log_std_linear2(log_std) + self.log_std_bias

        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)

        std = log_std.exp()

        dist = SquashedNormal(mean, std)

        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(dim=-1, keepdim=True)

        return sample, log_pi, dist.mean

    @torch.no_grad()
    def project_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, SIMBAV2Linear):
                module.project_weights()


class Actor(BasePolicy):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: SIMBAV2Config,
    ):
        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
        )

        self.log_std_bounds = config.log_std_bounds

        hidden_size = config.actor_config.hidden_size
        num_blocks = config.actor_config.num_blocks

        self.encoder = SIMBAV2Encoder(
            input_size=observation_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            c_shift=config.c_shift,
        )

        # Mean prediction head
        self.mean_linear1 = SIMBAV2Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.mean_scaler = SIMBAV2Scaler(size=hidden_size, init=1.0, scale=1.0)
        self.mean_linear2 = SIMBAV2Linear(
            in_features=hidden_size, out_features=num_actions
        )
        self.mean_bias = nn.Parameter(torch.zeros(num_actions))

        # Log standard-deviation prediction head
        self.log_std_linear1 = SIMBAV2Linear(
            in_features=hidden_size, out_features=hidden_size
        )
        self.log_std_scaler = SIMBAV2Scaler(size=hidden_size, init=1.0, scale=1.0)
        self.log_std_linear2 = SIMBAV2Linear(
            in_features=hidden_size, out_features=num_actions
        )
        self.log_std_bias = nn.Parameter(torch.zeros(num_actions))

    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self.encoder(state)

        mean = self.mean_linear1(features)
        mean = self.mean_scaler(mean)
        mean = self.mean_linear2(mean) + self.mean_bias

        log_std = self.log_std_linear1(features)
        log_std = self.log_std_scaler(log_std)
        log_std = self.log_std_linear2(log_std) + self.log_std_bias

        log_std = torch.tanh(log_std)

        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1.0)

        std = log_std.exp()

        dist = SquashedNormal(mean, std)

        sample = dist.rsample()
        log_pi = dist.log_prob(sample).sum(dim=-1, keepdim=True)

        return sample, log_pi, dist.mean

    @torch.no_grad()
    def project_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, SIMBAV2Linear):
                module.project_weights()
