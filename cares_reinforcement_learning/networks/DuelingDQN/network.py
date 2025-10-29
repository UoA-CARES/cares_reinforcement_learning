import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import DuelingDQNConfig
from cares_reinforcement_learning.networks.DQN import BaseNetwork


class BaseDuelingNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        feature_layer: nn.Module,
        value_stream: nn.Module,
        advantage_stream: nn.Module,
    ):
        super().__init__(observation_size=observation_size, num_actions=num_actions)

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
class DefaultNetwork(BaseDuelingNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [128, 128]
        value_stream_hidden_sizes = [128]
        advantage_stream_hidden_sizes = [128]

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
            observation_size=observation_size,
            num_actions=num_actions,
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )


class Network(BaseDuelingNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: DuelingDQNConfig,
    ):
        feature_layer = MLP(
            input_size=observation_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        value_stream = MLP(
            input_size=feature_layer.output_size,
            output_size=1,
            config=config.value_stream_config,
        )

        advantage_stream = MLP(
            input_size=feature_layer.output_size,
            output_size=num_actions,
            config=config.advantage_stream_config,
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )
