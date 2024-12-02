import torch
from batchrenorm import BatchRenorm1d
from torch import nn

from cares_reinforcement_learning.util.configurations import CrossQConfig


class BaseCritic(nn.Module):
    def __init__(self, Q1: nn.Module, Q2: nn.Module):
        super().__init__()

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = Q1

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = Q2

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2


class DefaultCritic(BaseCritic):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [2048, 2048]

        input_size = observation_size + num_actions

        # Q1 architecture
        # pylint: disable-next=invalid-name
        momentum = 0.01
        Q1 = nn.Sequential(
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
        Q2 = nn.Sequential(
            BatchRenorm1d(input_size, momentum=momentum),
            nn.Linear(input_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q1=Q1, Q2=Q2)


class Critic(BaseCritic):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: CrossQConfig,
    ):
        input_size = observation_size + num_actions
        hidden_sizes = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        momentum = 0.01
        Q1 = nn.Sequential(
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
        Q2 = nn.Sequential(
            BatchRenorm1d(input_size, momentum=momentum),
            nn.Linear(input_size, hidden_sizes[0], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[0], momentum=momentum),
            nn.Linear(hidden_sizes[0], hidden_sizes[1], bias=False),
            nn.ReLU(),
            BatchRenorm1d(hidden_sizes[1], momentum=momentum),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q1=Q1, Q2=Q2)
