from torch import nn

from cares_reinforcement_learning.networks.common import QNetwork, BaseCritic
from cares_reinforcement_learning.algorithm.configurations import SACLagConfig


class DefaultCostCritic(QNetwork):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        input_size = observation_size + num_actions

        # pylint: disable-next=non-parent-init-called
        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=1,
        )

        # Q architecture
        # pylint: disable-next=invalid-name
        self.Q = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )


class CostCritic(QNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: SACLagConfig):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size, output_size=1, config=config.cost_critic_config
        )
