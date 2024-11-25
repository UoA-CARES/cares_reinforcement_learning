import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import CTD4Config


class BaseCritic(nn.Module):
    def __init__(self, mean_layer: nn.Module, std_layer: nn.Module):
        super().__init__()

        self.mean_layer = mean_layer
        self.std_layer = std_layer

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        u = self.mean_layer(obs_action)
        std = self.std_layer(obs_action) + 1e-6
        return u, std


# This is the default base network for CTD4 for reference and testing of default network configurations
class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int, action_num: int):
        input_size = observation_size + action_num
        hidden_sizes = [256, 256]

        mean_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        std_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
            nn.Softplus(),
        )
        super().__init__(mean_layer=mean_layer, std_layer=std_layer)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, action_num: int, config: CTD4Config):
        input_size = observation_size + action_num
        hidden_sizes = config.hidden_size_critic

        mean_layer = MLP(
            input_size,
            hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        std_layer = MLP(
            input_size,
            hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
            output_activation_function=nn.Softplus,
        )

        super().__init__(mean_layer=mean_layer, std_layer=std_layer)
