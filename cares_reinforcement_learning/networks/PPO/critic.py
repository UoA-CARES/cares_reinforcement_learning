import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import PPOConfig


class BaseCritic(nn.Module):
    def __init__(self, Q1: nn.Module):
        super().__init__()

        self.Q1 = Q1

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q1 = self.Q1(state)
        return q1


class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int):
        hidden_sizes = [1024, 1024]

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q1=Q1)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, config: PPOConfig):
        hidden_sizes = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = MLP(
            observation_size,
            hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )
        super().__init__(Q1=Q1)
