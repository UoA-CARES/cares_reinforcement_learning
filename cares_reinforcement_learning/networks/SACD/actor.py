import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp import MLP
from cares_reinforcement_learning.util.configurations import SACDConfig


class BaseActor(nn.Module):
    def __init__(self, act_net: nn.Module, num_actions: int):
        super().__init__()
        self.network = act_net
        self.num_actions = num_actions
        self.keeper = None

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        logits = self.network(state)
        deterministic_action = torch.argmax(logits, dim=1)
        dist = torch.distributions.Categorical(logits=logits)
        sample_action = dist.sample()

        return sample_action, (dist.probs, torch.log(dist.probs)), deterministic_action
    

    def set_encoder(self, encoder: nn.Module) -> None:
        """Adds an encoder network to the actor."""
        self.encoder = encoder
        self.network = nn.Sequential(
            encoder,
            self.network,
        )

    
    def get_encoder(self) -> nn.Module:
        """Returns the encoder network of the actor."""
        return self.encoder
    

    def __call__(self, state: torch.Tensor) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        return super().__call__(state)


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
            num_actions=num_actions,
        )


class Actor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig):

        act_net = MLP(
            input_size=observation_size,
            output_size=num_actions,
            config=config.actor_config,
        )

        super().__init__(
            act_net=act_net,
            num_actions=num_actions,
        )
