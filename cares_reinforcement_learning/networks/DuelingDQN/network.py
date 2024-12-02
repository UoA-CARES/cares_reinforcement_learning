import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import DuelingDQNConfig


class BaseNetwork(nn.Module):
    def __init__(
        self,
        feature_layer: nn.Module,
        value_stream: nn.Module,
        advantage_stream: nn.Module,
    ):
        super().__init__()

        self.feature_layer = feature_layer
        self.value_stream = value_stream
        self.advantage_stream = advantage_stream

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        features = self.feature_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals


# This is the default base network for DuelingDQN for reference and testing of default network configurations
class DefaultNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [512, 512]
        value_stream_hidden_sizes = [512]
        advantage_stream_hidden_sizes = [512]

        feature_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        value_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], value_stream_hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(value_stream_hidden_sizes[0], 1),
        )

        advantage_stream = nn.Sequential(
            nn.Linear(hidden_sizes[1], advantage_stream_hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(advantage_stream_hidden_sizes[0], num_actions),
        )

        super().__init__(
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )


class Network(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: DuelingDQNConfig,
    ):
        hidden_sizes = config.feature_hidden_size
        value_stream_hidden_sizes = config.value_stream_hidden_size
        advantage_stream_hidden_sizes = config.advantage_stream_hidden_size

        feature_layer = MLP(
            observation_size,
            hidden_sizes,
            output_size=None,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        value_stream = MLP(
            hidden_sizes[-1],
            value_stream_hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        advantage_stream = MLP(
            hidden_sizes[-1],
            advantage_stream_hidden_sizes,
            output_size=num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        super().__init__(
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )
