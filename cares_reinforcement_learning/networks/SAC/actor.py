from torch import nn

from cares_reinforcement_learning.util.common import StochasticPolicy
from cares_reinforcement_learning.util.configurations import MLPConfig, SACConfig


class DefaultActor(StochasticPolicy):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
        log_std_bounds: list[float] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        if log_std_bounds is None:
            log_std_bounds = [-20.0, 2.0]

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            log_std_bounds=log_std_bounds,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        self.mean_linear = nn.Linear(hidden_sizes[-1], num_actions)
        self.log_std_linear = nn.Linear(hidden_sizes[-1], num_actions)


class Actor(StochasticPolicy):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SACConfig):

        log_std_bounds = config.log_std_bounds

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            log_std_bounds=log_std_bounds,
            config=config.actor_config,
        )
