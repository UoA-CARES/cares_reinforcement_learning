import torch
from torch import nn

from cares_reinforcement_learning.networks.common import MLP, MLPConfig, TrainableLayer, FunctionLayer
from cares_reinforcement_learning.util.configurations import SACDConfig


class BaseActor(nn.Module):
    def __init__(self, act_net: MLP, num_actions: int,  output_size: int = None):
        super().__init__()

        self.act_net = act_net

        output_size = output_size or act_net.output_size

        self.act_net.model.append(nn.Linear(output_size, num_actions))
        self.act_net.model.append(nn.Softmax(dim=-1))

        self.num_actions = num_actions

    def forward(
        self, state: torch.Tensor
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        action_probs = self.act_net(state)
        _, deterministic_actions = torch.topk(action_probs, k=self.num_actions)
        sampled_actions = torch.multinomial(action_probs, num_samples=1)

        # Offset any values which are zero by a small amount so no nan nonsense
        zero_offset = action_probs == 0.0
        zero_offset = zero_offset.float() * 1e-8
        log_action_probs = torch.log(action_probs + zero_offset)

        return sampled_actions, (action_probs, log_action_probs), deterministic_actions


class DefaultActor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int):
        hidden_sizes = [512, 512]

        act_net = MLP(
            input_size=observation_size,
            output_size=None,
            config=MLPConfig(
                layers=[
                    TrainableLayer(layer_type="Linear", out_features=512),
                    FunctionLayer(layer_type="ReLU"),
                    TrainableLayer(layer_type="Linear", in_features=512, out_features=512),
                    FunctionLayer(layer_type="ReLU"),
                ]
            )
        )

        super().__init__(
            act_net=act_net,
            num_actions=num_actions,
            output_size=hidden_sizes[-1],
        )


class Actor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, num_action_options: int, config: SACDConfig):

        act_net = MLP(
            input_size=observation_size,
            output_size=None,
            config=config.actor_config,
        )

        super().__init__(
            act_net=act_net,
            num_actions=num_actions,
        )
