import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import REDQConfig


class BaseCritic(nn.Module):
    def __init__(self, Q1: nn.Module):
        super().__init__()

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = Q1

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q = self.Q1(obs_action)
        return q


class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions
        hidden_sizes = [256, 256]

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        super().__init__(Q1=Q1)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: REDQConfig):
        input_size = observation_size + num_actions
        hidden_sizes = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = MLP(
            input_size,
            hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        super().__init__(Q1=Q1)
