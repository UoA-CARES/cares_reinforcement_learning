import torch
from torch import nn


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: list[int] | None = None,
    ):
        super().__init__()
        if hidden_size is None:
            hidden_size = [512, 512]

        self.num_actions = num_actions
        self.hidden_size = hidden_size

        self.act_net = nn.Sequential(
            nn.Linear(observation_size, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], num_actions),
            nn.Softmax(dim=-1),
        )

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        action_probs = self.act_net(state)
        max_probability_action = torch.argmax(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        # Offset any values which are zero by a small amount so no nan nonsense
        zero_offset = action_probs == 0.0
        zero_offset = zero_offset.float() * 1e-8
        log_action_probs = torch.log(action_probs + zero_offset)

        return action, (action_probs, log_action_probs), max_probability_action
