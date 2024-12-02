import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import PPOConfig


class BaseActor(nn.Module):
    def __init__(self, act_net: nn.Module, num_actions: int):
        super().__init__()

        self.num_actions = num_actions
        self.act_net = act_net

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.act_net(state)
        return output


class DefaultActor(BaseActor):
    def __init__(self, observation_size: int, num_actions: int):
        hidden_sizes = [1024, 1024]

        act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Tanh(),
        )

        super().__init__(act_net=act_net, num_actions=num_actions)


class Actor(BaseActor):
    def __init__(self, observation_size: int, num_actions: int, config: PPOConfig):

        hidden_sizes = config.hidden_size_actor

        act_net = MLP(
            observation_size,
            hidden_sizes,
            output_size=num_actions,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
            output_activation_function=nn.Tanh,
        )

        super().__init__(act_net=act_net, num_actions=num_actions)
