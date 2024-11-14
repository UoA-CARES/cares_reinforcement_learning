import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp


class Actor(nn.Module):
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

        self.act_net = nn.Sequential(
            nn.Linear(latent_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Tanh(),
        )
        self.apply(hlp.weight_init)

    def forward(
        self, state: torch.Tensor, detach_encoder: bool = False
    ) -> torch.Tensor:
        # NaSATD3 detatches the encoder at the output
        z_vector = self.encoder_net(state, detach_output=detach_encoder)
        output = self.act_net(z_vector)
        return output
