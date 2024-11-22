import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import SACDConfig


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig):
        super().__init__()

        self.hidden_sizes = config.hidden_size_critic
        self.num_actions = num_actions

        # Default network should have this architecture with hidden_sizes = [512, 512]:
        # self.QN = nn.Sequential(
        #     nn.Linear(observation_size, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[1], self.num_actions),
        # )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=self.num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=self.num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q1 = self.Q1(state)
        q2 = self.Q2(state)
        return q1, q2
