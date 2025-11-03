import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import PPOConfig


class BaseCritic(nn.Module):
    def __init__(self, Q: nn.Module):
        super().__init__()

        self.Q = Q

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q = self.Q(state)
        return q


class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int):
        hidden_sizes = [1024, 1024]

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q=Q)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, config: PPOConfig):
        # Q architecture
        # pylint: disable-next=invalid-name
        Q = MLP(
            input_size=observation_size,
            output_size=1,
            config=config.critic_config,
        )
        super().__init__(Q=Q)
