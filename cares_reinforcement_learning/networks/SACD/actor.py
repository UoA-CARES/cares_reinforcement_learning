import torch
from torch import nn

from cares_reinforcement_learning.util.common import MLP
from cares_reinforcement_learning.util.configurations import SACDConfig


class Actor(nn.Module):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SACDConfig):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.hidden_sizes = config.hidden_size_actor

        # Default actor network should have this architecture with hidden_sizes = [512, 512]:
        # self.act_net = nn.Sequential(
        #     nn.Linear(observation_size, self.hidden_sizes[0]),
        #     nn.ReLU(),
        #     nn.Linear(self.hidden_sizes[0], self.hidden_sizes[1]),
        #     nn.ReLU(),
        # )

        self.act_net = MLP(
            self.observation_size,
            self.hidden_sizes,
            output_size=None,
            norm_layer=config.norm_layer,
            norm_layer_args=config.norm_layer_args,
            hidden_activation_function=config.activation_function,
            hidden_activation_function_args=config.activation_function_args,
        )

        self.discrete_net = nn.Sequential(
            nn.Linear(self.hidden_sizes[-1], num_actions), nn.Softmax(dim=-1)
        )

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        action_probs = self.discrete_net(self.act_net(state))
        max_probability_action = torch.argmax(action_probs)
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()

        # Offset any values which are zero by a small amount so no nan nonsense
        zero_offset = action_probs == 0.0
        zero_offset = zero_offset.float() * 1e-8
        log_action_probs = torch.log(action_probs + zero_offset)

        return action, (action_probs, log_action_probs), max_probability_action
