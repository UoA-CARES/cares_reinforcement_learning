from torch import nn

from cares_reinforcement_learning.networks.common import (
    EnsembleCritic,
    QNetwork,
    BaseCritic,
)
from cares_reinforcement_learning.util.configurations import TQCConfig


class DefaultCritic(EnsembleCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        num_quantiles = 25
        num_critics = 5
        hidden_sizes = [512, 512, 512]

        # pylint: disable-next=non-parent-init-called
        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=num_quantiles,
        )

        self.critics: list[BaseCritic | nn.Sequential] = []

        for i in range(num_critics):
            critic_net = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], num_quantiles),
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.critics.append(critic_net)


class Critic(EnsembleCritic):
    def __init__(self, observation_size: int, num_actions: int, config: TQCConfig):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=config.num_quantiles,
            ensemble_size=config.num_critics,
            config=config.critic_config,
            critic_type=QNetwork,
        )
