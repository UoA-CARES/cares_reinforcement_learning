import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import TD3Config


class Critic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TD3Config):
        super().__init__()

        self.input_size = observation_size + num_actions
        self.hidden_size = config.hidden_size_critic

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = MLP(
            self.input_size,
            self.hidden_size,
            output_size=1,
            final_activation=None,
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = MLP(
            self.input_size,
            self.hidden_size,
            output_size=1,
            final_activation=None,
        )

    def forward(
        self, state: torch.Tensor, action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
