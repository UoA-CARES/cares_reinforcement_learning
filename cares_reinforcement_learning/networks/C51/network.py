import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import C51Config


class BaseNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        atom_size: int,
        v_min: float,
        v_max: float,
        network: MLP | nn.Sequential,
    ):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.atom_size = atom_size

        self.v_min = v_min
        self.v_max = v_max

        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size)

        self.network = network

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.network(state)

        q_atoms = output.view(-1, self.output_size, self.atom_size)
        dist = torch.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)

        q = torch.sum(dist * self.support, dim=2)

        return q, dist


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultNetwork(BaseNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [512, 512]
        atom_size = 51
        output_size = num_actions * atom_size

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
            input_size=observation_size,
            output_size=output_size,
            atom_size=atom_size,
            v_min=v_min,
            v_max=v_max,
            network=network,
        )


class Network(BaseNetwork):
    def __init__(self, observation_size: int, num_actions: int, config: C51Config):

        output_size = num_actions + config.num_atoms

        network = MLP(
            input_size=observation_size,
            output_size=output_size,
            config=config.network_config,
        )
        super().__init__(
            input_size=observation_size,
            output_size=output_size,
            atom_size=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            network=network,
        )
