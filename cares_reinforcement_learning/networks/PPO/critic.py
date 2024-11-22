import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import PPOConfig


class Critic(nn.Module):
    def __init__(self, observation_size: int, config: PPOConfig):
        super().__init__()

        self.hidden_sizes = config.hidden_size_critic

        # Default actor network should have this architecture with hidden_sizes = [1024, 1024]:
        # self.Q1 = nn.Sequential(
        #     nn.Linear(observation_size, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[1], 1),
        # )

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = MLP(
            observation_size,
            self.hidden_sizes,
            output_size=1,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        q1 = self.Q1(state)
        return q1
