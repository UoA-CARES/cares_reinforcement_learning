from torch import nn, Tensor

from cares_reinforcement_learning.networks.common import TwinQNetwork, BaseCritic
from cares_reinforcement_learning.util.configurations import SACConfig


# This is the default base network for TD3 for reference and testing of default network configurations
class DefaultCritic(TwinQNetwork):
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


class ResidualMLPBlock(nn.Module):
    def __init__(self, size: int, stride: int = 2):
        super().__init__()

        if stride < 2:
            raise ValueError("Stride must be greater than or equal to 2")

        self.resblock = nn.Sequential(
            nn.Linear(size, size),
            nn.ReLU(),
            nn.Linear(size, size),
        )

        for _ in range(stride - 2):
            self.resblock.append(nn.ReLU())
            self.resblock.append(nn.Linear(size, size))

        self.activation = nn.ReLU()

    def forward(self, x: Tensor) -> Tensor:
        return self.activation(self.resblock(x) + x)


class Critic(TwinQNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: SACConfig):
        input_size = observation_size + num_actions

        hidden_size = 256

        super().__init__(input_size=input_size, output_size=1, config=config.critic_config)

        if(False):
            # Q1 architecture
            # pylint: disable-next=invalid-name
            self.Q1 = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                ResidualMLPBlock(hidden_size),
                nn.Linear(hidden_size, 1),
            )

            # Q2 architecture
            # pylint: disable-next=invalid-name
            self.Q2 = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                ResidualMLPBlock(hidden_size),
                nn.Linear(hidden_size, 1),

        )
