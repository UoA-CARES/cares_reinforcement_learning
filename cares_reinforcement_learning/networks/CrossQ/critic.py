from torch import nn

from cares_reinforcement_learning.networks.batchrenorm import BatchRenorm1d
from cares_reinforcement_learning.networks.common import TwinQNetwork
from cares_reinforcement_learning.util.configurations import CrossQConfig, MLPConfig


class DefaultCritic(TwinQNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [2048, 2048]

        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=1,
            config=MLPConfig(hidden_sizes=hidden_sizes),
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        momentum = 0.01
        self.Q1 = nn.Sequential(
            BatchRenorm1d(input_size, momentum=momentum),
            nn.Linear(input_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            BatchRenorm1d(input_size, momentum=momentum),
            nn.Linear(input_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1),
        )


class Critic(TwinQNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: CrossQConfig):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size, output_size=1, config=config.critic_config
        )