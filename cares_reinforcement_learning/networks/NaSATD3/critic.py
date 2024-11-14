import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp


class Critic(nn.Module):
    def __init__(
        self,
        latent_size: int,
        num_actions: int,
        encoder: nn.Module,
        hidden_size: list[int] = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [1024, 1024]

        self.encoder_net = encoder
        self.hidden_size = hidden_size

        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(latent_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(latent_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        self.apply(hlp.weight_init)

    def forward(
        self, state: torch.Tensor, action: torch.Tensor, detach_encoder: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # NaSATD3 detatches the encoder at the output
        z_vector = self.encoder_net(state, detach_output=detach_encoder)
        obs_action = torch.cat([z_vector, action], dim=1)
        q1 = self.Q1(obs_action)
        q2 = self.Q2(obs_action)
        return q1, q2
