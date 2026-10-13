from torch import nn

from cares_reinforcement_learning.algorithm.configurations import (
    FunctionLayer,
    NormLayer,
    ResidualLayer,
    SIMBAConfig,
    TrainableLayer,
)
from cares_reinforcement_learning.networks.common import BaseCritic, TwinQNetwork
from cares_reinforcement_learning.networks.mlp_architecture import MLPResidualBlock


class DefaultCritic(TwinQNetwork):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: int = 512,
    ):
        input_size = observation_size + num_actions

        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=1,
        )

        residual = ResidualLayer(
            main_layers=[
                NormLayer(layer_type="LayerNorm"),
                TrainableLayer(
                    layer_type="Linear",
                    out_features=hidden_size * 4,
                ),
                FunctionLayer(layer_type="ReLU"),
                TrainableLayer(
                    layer_type="Linear",
                    out_features=hidden_size,
                ),
            ]
        )

        def make_q_network() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_size, hidden_size),
                MLPResidualBlock(hidden_size, residual),
                MLPResidualBlock(hidden_size, residual),
                nn.LayerNorm(hidden_size),
                nn.Linear(hidden_size, 1),
            )

        self.Q1 = make_q_network()
        self.Q2 = make_q_network()


class Critic(TwinQNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: SIMBAConfig,
    ):
        super().__init__(
            input_size=observation_size + num_actions,
            output_size=1,
            config=config.critic_config,
        )
