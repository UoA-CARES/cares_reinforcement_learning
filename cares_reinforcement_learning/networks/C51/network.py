import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.networks.DQN import BaseNetwork
from cares_reinforcement_learning.util.configurations import C51Config


class BaseC51Network(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        output_size: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        network: MLP | nn.Sequential,
    ):
        super().__init__(observation_size=observation_size, num_actions=num_actions)

        self.output_size = output_size
        self.num_atoms = num_atoms

        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(self.v_min, self.v_max, self.num_atoms)

        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        dist = self.dist(state)

        self.support = self.support.to(dist.device)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)

        out_dim = int(self.output_size / self.num_atoms)
        q_atoms = output.view(-1, out_dim, self.num_atoms)

        dist = torch.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseC51Network):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [64, 64]
        num_atoms = 51
        output_size = num_actions * num_atoms

        v_min = 0.0
        v_max = 200.0

        network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], output_size),
        )
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            output_size=output_size,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            network=network,
        )


class Network(BaseC51Network):
    def __init__(self, observation_size: int, num_actions: int, config: C51Config):

        output_size = num_actions * config.num_atoms

        network = MLP(
            input_size=observation_size,
            output_size=output_size,
            config=config.network_config,
        )
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            output_size=output_size,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            network=network,
        )
