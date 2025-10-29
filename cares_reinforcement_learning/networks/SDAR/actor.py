import torch
from torch import nn

from cares_reinforcement_learning.networks.common import (
    MLP,
    BasePolicy,
    TanhGaussianPolicy,
)
from cares_reinforcement_learning.util.configurations import (
    FunctionLayer,
    MLPConfig,
    SDARConfig,
    TrainableLayer,
)


class BaseActor(BasePolicy):

    def __init__(
        self,
        input_size: int,
        num_actions: int,
        selector_net: nn.Module,
        actor_net: TanhGaussianPolicy,
    ):
        super().__init__(input_size=input_size, num_actions=num_actions)

        self.selector_net = selector_net
        self.actor_net = actor_net

        self.dummy_action = -2.0

    # pylint: disable-next=arguments-differ, arguments-renamed
    def forward(  # type: ignore[override]
        self,
        state: torch.Tensor,
        prev_action: torch.Tensor,
        force_act: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        # Stage 1: Selection Policy β
        if force_act:
            binary_mask = torch.ones_like(prev_action, dtype=torch.float32)
            act_probs = torch.ones_like(prev_action, dtype=torch.float32)
        else:
            selector_input = torch.cat([state, prev_action], dim=-1)
            act_probs = torch.sigmoid(self.selector_net(selector_input))
            act_probs = act_probs.clamp(min=1e-6, max=1 - 1e-6)
            binary_mask = torch.bernoulli(act_probs)

        # Compute log_prob of b under Bernoulli distribution
        bernoulli = torch.distributions.Bernoulli(probs=act_probs)

        log_beta = bernoulli.log_prob(binary_mask).sum(dim=-1, keepdim=True)

        # Stage 2: Action Policy π
        dummy_action = torch.full_like(prev_action, self.dummy_action)
        a_mix = (1 - binary_mask) * prev_action + binary_mask * dummy_action
        actor_input = torch.cat([state, a_mix], dim=-1)

        sample, log_pi, mean = self.actor_net(actor_input)

        # Final action a = (1 - b) * a_prev + b * a_new
        sample_action = (1 - binary_mask) * prev_action + binary_mask * sample

        # Use mean instead of sample
        mean_action = (1 - binary_mask) * prev_action + binary_mask * mean

        return (
            sample_action,
            log_pi,
            mean_action,
            act_probs,
            binary_mask,
            log_beta,
        )


class DefaultActor(BaseActor):
    """Default Actor class for SDAR, using a simple MLP policy."""

    def __init__(self, observation_size: int, num_actions):

        input_size = observation_size + num_actions

        hidden_sizes = [256, 256]
        log_std_bounds = [-20.0, 2.0]

        selector_net = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
            nn.Sigmoid(),
        )

        actor_config: MLPConfig = MLPConfig(
            layers=[
                TrainableLayer(layer_type="Linear", out_features=256),
                FunctionLayer(layer_type="ReLU"),
                TrainableLayer(layer_type="Linear", in_features=256, out_features=256),
                FunctionLayer(layer_type="ReLU"),
            ]
        )

        actor_net = TanhGaussianPolicy(
            input_size=input_size,
            num_actions=num_actions,
            log_std_bounds=log_std_bounds,
            config=actor_config,
        )

        super().__init__(
            input_size=input_size,
            num_actions=num_actions,
            selector_net=selector_net,
            actor_net=actor_net,
        )


class Actor(BaseActor):
    # DiagGaussianActor
    """torch.distributions implementation of an diagonal Gaussian policy."""

    def __init__(self, observation_size: int, num_actions: int, config: SDARConfig):

        input_size = observation_size + num_actions

        # Selection Network: β(b | s, a_prev)
        selector_net = MLP(
            input_size=input_size,
            output_size=num_actions,
            config=config.selector_config,
        )

        # Action Network: π(â | s, a_mix)
        actor_net = TanhGaussianPolicy(
            input_size=input_size,
            num_actions=num_actions,
            log_std_bounds=config.log_std_bounds,
            config=config.actor_config,
        )

        super().__init__(
            input_size=input_size,
            num_actions=num_actions,
            selector_net=selector_net,
            actor_net=actor_net,
        )
