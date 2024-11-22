import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import DQNConfig


class Network(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: DQNConfig):
        super().__init__()
        self.hidden_sizes = config.hidden_size

        # Default network should have this architecture with hidden_sizes = [512, 512]:
        # self.h_linear_1 = nn.Linear(
        #     in_features=observation_size, out_features=self.hidden_size[0]
        # )
        # self.h_linear_2 = nn.Linear(
        #     in_features=self.hidden_size[0], out_features=self.hidden_size[1]
        # )
        # self.h_linear_3 = nn.Linear(
        #     in_features=self.hidden_size[1], out_features=num_actions
        # )
        self.network = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)
        return output
