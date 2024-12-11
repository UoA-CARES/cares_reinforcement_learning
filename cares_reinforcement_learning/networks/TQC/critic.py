from torch import nn

from cares_reinforcement_learning.networks.common import EnsembleCritic
from cares_reinforcement_learning.util.configurations import MLPConfig, TQCConfig


class DefaultCritic(EnsembleCritic):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        num_quantiles = 25
        num_critics = 5
        hidden_sizes = [512, 512, 512]

        super().__init__(
            input_size=input_size,
            output_size=num_quantiles,
            ensemble_size=num_critics,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

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
            self.critics[i] = critic_net


class Critic(EnsembleCritic):
    def __init__(self, observation_size: int, num_actions: int, config: TQCConfig):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=config.num_quantiles,
            ensemble_size=config.num_critics,
            config=config.critic_config,
        )
