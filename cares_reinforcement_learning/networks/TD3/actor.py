from torch import nn

from cares_reinforcement_learning.util.common import DeterministicPolicy
from cares_reinforcement_learning.util.configurations import MLPConfig, TD3Config


class DefaultActor(DeterministicPolicy):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            config=MLPConfig(
                hidden_sizes=hidden_sizes, output_activation_function=nn.Tanh.__name__
            ),
        )

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Tanh(),
        )


class Actor(DeterministicPolicy):
    def __init__(self, observation_size: int, num_actions: int, config: TD3Config):

        super().__init__(
            input_size=observation_size,
            num_actions=num_actions,
            config=config.actor_config,
        )
