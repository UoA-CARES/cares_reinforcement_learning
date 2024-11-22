import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import TQCConfig


# TODO create the ensemble of critics in TQC - reduce this to just one critic
class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TQCConfig):
        super().__init__()

        self.input_size = observation_size + num_actions

        self.q_networks = []
        self.num_quantiles = config.num_quantiles
        self.num_critics = config.num_critics
        self.hidden_sizes = config.hidden_size_critic

        # Default critic network should have this architecture with hidden_sizes = [512, 512, 512]:
        # critic_net = nn.Sequential(
        #     nn.Linear(observation_size + num_actions, self.hidden_size[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size[0], self.hidden_size[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size[1], self.hidden_size[2]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_size[2], num_quantiles),
        # )

        for i in range(self.num_critics):
            # critic_net = MLP(
            #     observation_size + num_actions, self.hidden_sizes, self.num_quantiles
            # )
            critic_net = MLP(
                self.input_size,
                self.hidden_sizes,
                output_size=self.num_quantiles,
                norm_layer=config.norm_layer,
                norm_layer_args=config.norm_layer_args,
                hidden_activation_function=config.activation_function,
                hidden_activation_function_args=config.activation_function_args,
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.q_networks.append(critic_net)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        network_input = torch.cat((state, action), dim=1)
        quantiles = torch.stack(
            tuple(critic(network_input) for critic in self.q_networks), dim=1
        )
        return quantiles
