import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import TD3Config


class Actor(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TD3Config):
        super().__init__()

        self.num_actions = num_actions
        self.hidden_sizes = config.hidden_size_actor

        self.act_net = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=num_actions,
            norm_layer_parameters=config.norm_layer,
            activation_function_parameters=config.activation_function,
            final_activation_parameters=(nn.Tanh, {}),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output
