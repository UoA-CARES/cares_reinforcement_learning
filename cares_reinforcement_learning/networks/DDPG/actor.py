import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import DDPGConfig


class Actor(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: DDPGConfig):
        super().__init__()

        self.num_actions = num_actions
        self.hidden_sizes = config.hidden_size_actor

        # Default actor network should have this architecture with hidden_sizes = [1024, 1024]:
        # self.act_net = nn.Sequential(
        #     nn.Linear(observation_size, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[1], num_actions),
        #     nn.Tanh(),
        # )

        self.act_net = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
            output_activation_function=nn.Tanh,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output
