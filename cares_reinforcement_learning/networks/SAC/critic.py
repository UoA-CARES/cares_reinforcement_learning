import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import SACConfig


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: SACConfig):
        super().__init__()

        self.input_size = observation_size + num_actions
        self.hidden_sizes = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = MLP(
            self.input_size,
            self.hidden_sizes,
            output_size=1,
            norm_layer_parameters=config.norm_layer,
            activation_function_parameters=config.activation_function,
            final_activation_parameters=None,
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = MLP(
            self.input_size,
            self.hidden_sizes,
            output_size=1,
            norm_layer_parameters=config.norm_layer,
            activation_function_parameters=config.activation_function,
            final_activation_parameters=None,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
