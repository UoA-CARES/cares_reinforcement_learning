import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP, TwinQNetwork, BaseCritic
from cares_reinforcement_learning.util.configurations import SACDConfig


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

        input_size = observation_size

        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=num_actions,
        )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )


class Critic(TwinQNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig):
        input_size = observation_size

        super().__init__(
            input_size=input_size, output_size=num_actions, config=config.critic_config
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2
