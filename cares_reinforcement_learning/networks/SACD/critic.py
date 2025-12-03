import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import SACDConfig


class BaseCritic(nn.Module):
    def __init__(self, Q1: nn.Module, Q2: nn.Module, encoder_net: MLP):
        super().__init__()
        self.Q1 = Q1
        self.Q2 = Q2
        if encoder_net is None:
            encoder_net = nn.Identity()
        self.encoder_net = encoder_net

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded_state = self.encoder_net(state)
        q1 = self.Q1(encoded_state)
        q2 = self.Q2(encoded_state)
        return q1, q2


class DefaultCritic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int):

        hidden_sizes = [512, 512]

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        Q2 = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

        super().__init__(Q1=Q1, Q2=Q2)


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig, encoder_net: MLP):

        # Q1 architecture
        # pylint: disable-next=invalid-name
        Q1 = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.critic_config,
        )
        # Q2 architecture
        # pylint: disable-next=invalid-name
        Q2 = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.critic_config,
        )

        super().__init__(Q1=Q1, Q2=Q2, encoder_net=encoder_net)
