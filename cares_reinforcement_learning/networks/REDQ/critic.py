from torch import nn

from cares_reinforcement_learning.networks.common import EnsembleCritic
from cares_reinforcement_learning.util.configurations import MLPConfig, REDQConfig


class DefaultCritic(EnsembleCritic):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions

        ensemble_size = 10
        hidden_sizes = [256, 256]

        super().__init__(
            input_size=input_size,
            output_size=1,
            ensemble_size=ensemble_size,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        for i in range(ensemble_size):
            critic_net = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], 1),
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.critics[i] = critic_net


class Critic(EnsembleCritic):
    def __init__(self, observation_size: int, num_actions: int, config: REDQConfig):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=1,
            ensemble_size=config.ensemble_size,
            config=config.critic_config,
        )
