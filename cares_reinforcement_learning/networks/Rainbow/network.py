import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP, NoisyLinear
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.util.configurations import RainbowConfig


class BaseRainbowNetwork(BaseNetwork):
    def __init__(
        self,
        oberservation_size: int,
        num_actions: int,
        output_size: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        feature_layer: MLP | nn.Sequential,
        value_stream: MLP | nn.Sequential,
        advantage_stream: MLP | nn.Sequential,
    ):
        super().__init__(observation_size=oberservation_size, num_actions=num_actions)

        self.output_size = output_size
        self.num_atoms = num_atoms

        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        self.feature_layer = feature_layer
        self.value_stream = value_stream
        self.advantage_stream = advantage_stream

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.dist(state)

        self.support = self.support.to(dist.device)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, state: torch.Tensor) -> torch.Tensor:
        feature = self.feature_layer(state)

        value = self.value_stream(feature)
        advantage = self.advantage_stream(feature)

        out_dim = int(self.output_size / self.num_atoms)
        advantage = advantage.view(-1, out_dim, self.num_atoms)
        value = value.view(-1, 1, self.num_atoms)

        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = torch.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        for module in self.feature_layer.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

        for module in self.value_stream.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()

        for module in self.advantage_stream.modules():
            if hasattr(module, "reset_noise"):
                module.reset_noise()


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseRainbowNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [128]
        value_stream_hidden_sizes = [128]
        advantage_stream_hidden_sizes = [128]

        num_atoms = 51
        output_size = num_actions * num_atoms

        v_min = 0.0
        v_max = 200.0

        feature_layer = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
        )

        value_stream = nn.Sequential(
            NoisyLinear(hidden_sizes[0], value_stream_hidden_sizes[0]),
            nn.ReLU(),
            NoisyLinear(value_stream_hidden_sizes[0], num_atoms),
        )

        advantage_stream = nn.Sequential(
            NoisyLinear(hidden_sizes[0], advantage_stream_hidden_sizes[0]),
            nn.ReLU(),
            NoisyLinear(advantage_stream_hidden_sizes[0], output_size),
        )

        super().__init__(
            oberservation_size=observation_size,
            num_actions=num_actions,
            output_size=output_size,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )


class Network(BaseRainbowNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: RainbowConfig):

        output_size = num_actions * config.num_atoms

        feature_layer = MLP(
            input_size=observation_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        value_stream = MLP(
            input_size=feature_layer.output_size,
            output_size=config.num_atoms,
            config=config.value_stream_config,
        )

        advantage_stream = MLP(
            input_size=feature_layer.output_size,
            output_size=output_size,
            config=config.advantage_stream_config,
        )

        super().__init__(
            oberservation_size=observation_size,
            num_actions=num_actions,
            output_size=output_size,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            feature_layer=feature_layer,
            value_stream=value_stream,
            advantage_stream=advantage_stream,
        )
