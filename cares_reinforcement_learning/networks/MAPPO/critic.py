import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.algorithm.configurations import MAPPOConfig


class BaseCritic(nn.Module):
    def __init__(self, Q: nn.Module):
        super().__init__()

        self.Q = Q

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q = self.Q(state)
        return q


class DefaultCritic(BaseCritic):
    def __init__(
        self,
        observation_size: dict,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        input_size = observation_size["state"]
        output_size = len(observation_size.keys())

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )

        super().__init__(Q=Q)


class Critic(BaseCritic):
    def __init__(self, observation_size: dict, config: MAPPOConfig):
        # Q architecture
        input_size = observation_size["state"]
        output_size = len(observation_size.keys())

        # pylint: disable-next=invalid-name
        Q = MLP(
            input_size=input_size,
            output_size=output_size,
            config=config.critic_config,
        )
        super().__init__(Q=Q)
