from torch import nn

from cares_reinforcement_learning.util.common import TwinQValueCritic
from cares_reinforcement_learning.util.configurations import TD3Config, MLPConfig


# This is the default base network for TD3 for reference and testing of default network configurations
class DefaultCritic(TwinQValueCritic):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=1,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )


class Critic(TwinQValueCritic):
    def __init__(self, observation_size: int, num_actions: int, config: TD3Config):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size, output_size=1, config=config.critic_config
        )
