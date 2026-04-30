import torch
from torch import nn

from cares_reinforcement_learning.networks.mlp_architecture import MLP
from cares_reinforcement_learning.algorithm.configurations import SACDConfig


class BaseActor(nn.Module):
    def __init__(self, act_net: nn.Module, discretisation_input_size: int, num_actions: int):
        super().__init__()

        self.act_net = act_net
        self.num_actions = num_actions

        self.discrete_net = nn.Sequential(
            nn.Linear(discretisation_input_size, num_actions), 
            nn.Softmax(dim=-1)
        )


    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        action_probs = self.discrete_net(self.act_net(state))
        most_probable_action = torch.argmax(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        sample_action = dist.sample()

        # Offset any values which are zero by a small amount so no nan nonsense
        zero_offset = action_probs == 0.0
        zero_offset = zero_offset.float() * 1e-8
        log_action_probs = torch.log(action_probs + zero_offset)

        return sample_action, (action_probs, log_action_probs), most_probable_action
    
    
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
            discretisation_input_size=hidden_sizes[-1],
            num_actions=num_actions,
        )


class Actor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig):

        act_net = MLP(
            input_size=observation_size,
            output_size=None,
            config=config.actor_config,
        )

        discretisation_input = act_net.output_size

        super().__init__(
            act_net=act_net,
            discretisation_input_size=discretisation_input,
            num_actions=num_actions,
        )
