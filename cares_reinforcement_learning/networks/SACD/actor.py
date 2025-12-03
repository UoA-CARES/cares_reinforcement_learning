import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import SACDConfig


class BaseActor(nn.Module):
    def __init__(self, act_net: nn.Module, num_actions: int, encoder_net: MLP):
        super().__init__()
        if encoder_net is None:
            encoder_net = nn.Identity()
        self.network = nn.Sequential(
            encoder_net,
            act_net,
        )
        self.num_actions = num_actions
        self.keeper = None

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        logits = self.network(state)
        deterministic_action = torch.argmax(logits, dim=1)
        dist = torch.distributions.Categorical(logits=logits)
        sample_action = dist.sample()

        # Offset any values which are zero by a small amount so no nan nonsense
        zero_offset = logits == 0.0
        zero_offset = zero_offset.float() * 1e-8
        log_logits = torch.log(logits + zero_offset)

        return sample_action, (dist.probs, dist.logits), deterministic_action


class DefaultActor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int):
        hidden_sizes = [512, 512]

        act_net = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
        )

        super().__init__(
            act_net=act_net,
            encoder_net=None,
            num_actions=num_actions,
        )


class Actor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig, encoder_net: MLP):

        act_net = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.actor_config,
        )

        super().__init__(
            act_net=act_net,
            num_actions=num_actions,
            encoder_net=encoder_net,
        )
