import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import DuelingDQNConfig


class Network(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: DuelingDQNConfig,
    ):
        super().__init__()

        self.hidden_sizes = config.feature_hidden_size
        self.value_stream_hidden_sizes = config.value_stream_hidden_size
        self.advantage_stream_hidden_sizes = config.advantage_stream_hidden_size

        self.observation_size = observation_size
        self.action_num = num_actions

        # Default network should have this architecture with hidden_sizes = [512, 512]:
        # self.feature_layer = nn.Sequential(
        #     nn.Linear(self.observation_size, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        # )

        self.feature_layer = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=None,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        # Default network should have this architecture with hidden_sizes = [512]:
        # self.value_stream = nn.Sequential(
        #     nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[2], 1),
        # )
        self.value_stream = MLP(
            self.hidden_sizes[-1],
            self.value_stream_hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        # Default network should have this architecture with hidden_sizes = [512]:
        # self.advantage_stream = nn.Sequential(
        #     nn.Linear(self.hidden_sizes[1], self.hidden_sizes[2]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[2], self.action_num),
        # )
        self.advantage_stream = MLP(
            self.hidden_sizes[-1],
            self.advantage_stream_hidden_sizes,
            output_size=self.action_num,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals
