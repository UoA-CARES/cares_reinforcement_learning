import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import TQCConfig


class BaseCritic(nn.Module):
    def __init__(self, critic_nets: list[nn.Sequential] | list[MLP]):
        super().__init__()

        self.q_networks = []

        for i, critic_net in enumerate(critic_nets):
            self.add_module(f"critic_net_{i}", critic_net)
            self.q_networks.append(critic_net)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        network_input = torch.cat((state, action), dim=1)
        quantiles = torch.stack(
            tuple(critic(network_input) for critic in self.q_networks), dim=1
        )
        return quantiles


class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions

        critic_nets = []
        num_quantiles = 25
        num_critics = 5
        hidden_sizes = [512, 512, 512]

        for _ in range(num_critics):
            critic_net = nn.Sequential(
                nn.Linear(input_size, hidden_sizes[0]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[0], hidden_sizes[1]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[1], hidden_sizes[2]),
                nn.ReLU(),
                nn.Linear(hidden_sizes[2], num_quantiles),
            )
            critic_nets.append(critic_net)

        super().__init__(critic_nets)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: TQCConfig):
        input_size = observation_size + num_actions

        critic_nets = []
        num_quantiles = config.num_quantiles
        num_critics = config.num_critics
        hidden_sizes = config.hidden_size_critic

        for _ in range(num_critics):
            critic_net = MLP(
                input_size,
                hidden_sizes,
                output_size=num_quantiles,
                norm_layer=config.norm_layer,
                norm_layer_args=config.norm_layer_args,
                hidden_activation_function=config.activation_function,
                hidden_activation_function_args=config.activation_function_args,
            )
            critic_nets.append(critic_net)

        super().__init__(critic_nets)
