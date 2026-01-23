import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp import MLP
from cares_reinforcement_learning.util.configurations import SACDConfig


class BaseActor(nn.Module):
    def __init__(self, act_net: nn.Module, num_actions: int):
        super().__init__()
        self.network = act_net
        self.num_actions = num_actions

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
    

    def enable_film(self, num_tasks: int) -> None:
        self.film_layers = self.network.film_layers
        self.film_fc_layer = nn.Linear(num_tasks, len(self.film_layers) * 2).cuda()


    def update_film_params(self, tasks: torch.Tensor) -> torch.Tensor:
        # Assume tasks is of shape (batch_size, num_tasks)
        film_params = self.film_fc_layer(tasks)
        for i, film_layer in enumerate(self.film_layers):
            scales = film_params[:, 2 * i]
            shifts = film_params[:, 2 * i + 1]
            film_layer.set_film_parameters(scales, shifts)
    

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
