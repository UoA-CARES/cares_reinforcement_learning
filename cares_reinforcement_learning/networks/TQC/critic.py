import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import TQCConfig


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TQCConfig):
        super().__init__()

        self.q_networks = []
        self.num_quantiles = config.num_quantiles
        self.num_critics = config.num_critics
        self.hidden_sizes = config.hidden_size_critic

        for i in range(self.num_critics):
            critic_net = MLP(
                observation_size + num_actions, self.hidden_sizes, self.num_quantiles
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.q_networks.append(critic_net)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        network_input = torch.cat((state, action), dim=1)
        quantiles = torch.stack(
            tuple(critic(network_input) for critic in self.q_networks), dim=1
        )
        return quantiles
