import torch
import torch.nn.functional as F
from torch import nn

from cares_reinforcement_learning.algorithm.configurations import SIMBAV2Config
from cares_reinforcement_learning.networks.common import BaseCritic
from cares_reinforcement_learning.networks.simba_architecture import (
    SIMBAV2Encoder,
    SIMBAV2Linear,
    SIMBAV2Scaler,
)


class SIMBAV2CategoricalQNetwork(nn.Module):
    """
    Single SimbaV2 categorical Q-network.

    Structure:
        [observation, action]
        -> SimbaV2 encoder
        -> Linear
        -> Scaler
        -> Linear
        -> categorical return distribution

    The scalar Q-value is the expectation of the categorical
    return distribution.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_blocks: int,
        num_atoms: int,
        v_min: float,
        v_max: float,
        c_shift: float = 3.0,
    ):
        super().__init__()

        self.encoder = SIMBAV2Encoder(
            input_size=input_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            c_shift=c_shift,
        )

        self.value_linear1 = SIMBAV2Linear(
            in_features=hidden_size,
            out_features=hidden_size,
        )

        self.value_scaler = SIMBAV2Scaler(
            size=hidden_size,
            init=1.0,
            scale=1.0,
        )

        self.value_linear2 = SIMBAV2Linear(
            in_features=hidden_size,
            out_features=num_atoms,
        )

        self.value_bias = nn.Parameter(torch.zeros(num_atoms))

        self.register_buffer(
            "support",
            torch.linspace(
                v_min,
                v_max,
                num_atoms,
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        input_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(input_value)

        logits = self.value_linear1(features)
        logits = self.value_scaler(logits)
        logits = self.value_linear2(logits)
        logits = logits + self.value_bias

        log_probs = F.log_softmax(
            logits,
            dim=-1,
        )

        probabilities = log_probs.exp()

        q_value = torch.sum(
            probabilities * self.support,
            dim=-1,
            keepdim=True,
        )

        return q_value, log_probs


class DefaultCritic(BaseCritic):
    """
    Default SimbaV2 twin categorical critic.

    Reference architecture:
        hidden size: 512
        blocks: 2
        atoms: 101
        support: [-5, 5]
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_size: int = 512,
        num_blocks: int = 2,
        num_atoms: int = 101,
        v_min: float = -5.0,
        v_max: float = 5.0,
        c_shift: float = 3.0,
    ):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=1,
        )

        # pylint: disable=invalid-name
        self.Q1 = SIMBAV2CategoricalQNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            c_shift=c_shift,
        )

        self.Q2 = SIMBAV2CategoricalQNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_atoms=num_atoms,
            v_min=v_min,
            v_max=v_max,
            c_shift=c_shift,
        )

    @property
    def support(self) -> torch.Tensor:
        return self.Q1.support

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, _, q2, _ = self.forward_distribution(
            state=state,
            action=action,
        )

        return q1, q2

    def forward_distribution(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        observation_action = torch.cat(
            (state, action),
            dim=-1,
        )

        q1, log_probs1 = self.Q1(observation_action)
        q2, log_probs2 = self.Q2(observation_action)

        return q1, log_probs1, q2, log_probs2

    @torch.no_grad()
    def project_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, SIMBAV2Linear):
                module.project_weights()


class Critic(BaseCritic):
    """
    Configurable SimbaV2 twin categorical critic.
    """

    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        config: SIMBAV2Config,
    ):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=1,
        )

        hidden_size = config.critic_config.hidden_size
        num_blocks = config.critic_config.num_blocks

        # pylint: disable=invalid-name
        self.Q1 = SIMBAV2CategoricalQNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            c_shift=config.c_shift,
        )

        self.Q2 = SIMBAV2CategoricalQNetwork(
            input_size=input_size,
            hidden_size=hidden_size,
            num_blocks=num_blocks,
            num_atoms=config.num_atoms,
            v_min=config.v_min,
            v_max=config.v_max,
            c_shift=config.c_shift,
        )

    @property
    def support(self) -> torch.Tensor:
        return self.Q1.support

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        q1, _, q2, _ = self.forward_distribution(
            state=state,
            action=action,
        )

        return q1, q2

    def forward_distribution(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        observation_action = torch.cat(
            (state, action),
            dim=-1,
        )

        q1, log_probs1 = self.Q1(observation_action)
        q2, log_probs2 = self.Q2(observation_action)

        return q1, log_probs1, q2, log_probs2

    @torch.no_grad()
    def project_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, SIMBAV2Linear):
                module.project_weights()
