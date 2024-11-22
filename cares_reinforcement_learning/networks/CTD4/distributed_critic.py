import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import CTD4Config


class DistributedCritic(nn.Module):
    def __init__(self, observation_size: int, action_num: int, config: CTD4Config):
        super().__init__()

        self.input_size = observation_size + action_num
        self.hidden_sizes = config.hidden_size_critic

        # Default critic network should have this architecture with hidden_sizes = [256, 256]:
        # self.mean_layer = nn.Sequential(
        #     nn.Linear(observation_size + action_num, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[1], 1),
        # )

        self.mean_layer = MLP(
            self.input_size,
            self.hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        # Default critic network should have this architecture with hidden_sizes = [256, 256]:
        # self.std_layer = nn.Sequential(
        #     nn.Linear(observation_size + action_num, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[1], 1),
        #     nn.Softplus(),
        # )

        self.std_layer = MLP(
            self.input_size,
            self.hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
            output_activation_function=nn.Softplus,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        u = self.mean_layer(obs_action)
        std = self.std_layer(obs_action) + 1e-6
        return u, std
