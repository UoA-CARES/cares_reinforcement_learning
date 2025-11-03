import torch
from torch import nn

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import TD7Config


class BaseCritic(nn.Module):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):
        super().__init__()

        self.input_size = observation_size + num_actions
        self.output_size = 1

        self.feature_layer_one: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        self.q_input_size = 2 * config.zs_dim + self.feature_layer_one.output_size

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1: MLP | nn.Sequential = MLP(
            input_size=self.q_input_size,
            output_size=1,
            config=config.critic_config,
        )

        self.feature_layer_two: MLP | nn.Sequential = MLP(
            input_size=self.input_size,
            output_size=None,
            config=config.feature_layer_config,
        )

        self.q_input_size = 2 * config.zs_dim + self.feature_layer_two.output_size

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2: MLP | nn.Sequential = MLP(
            input_size=self.q_input_size,
            output_size=1,
            config=config.critic_config,
        )

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        zsa: torch.Tensor,
        zs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_action = torch.cat([state, action], dim=1)
        embeddings = torch.cat([zsa, zs], dim=1)

        q1 = self.feature_layer_one(obs_action)
        q1 = hlp.avg_l1_norm(q1)
        q1 = torch.cat([q1, embeddings], dim=1)
        q1 = self.Q1(q1)

        q2 = self.feature_layer_two(obs_action)
        q2 = hlp.avg_l1_norm(q2)
        q2 = torch.cat([q2, embeddings], dim=1)
        q2 = self.Q2(q2)

        return q1, q2


# This is the default base network for TD7 for reference and testing of default network configurations
class DefaultCritic(BaseCritic):
    # pylint: disable=super-init-not-called
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        hidden_sizes: list[int] | None = None,
    ):
        if hidden_sizes is None:
            hidden_sizes = [256, 256]

        input_size = observation_size + num_actions
        zs_dim = 256

        # pylint: disable-next=non-parent-init-called
        nn.Module.__init__(self)

        self.feature_layer_one = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
        )

        self.q_input_size = 2 * zs_dim + hidden_sizes[0]

        # Q1 architecture
        # pylint: disable-next=invalid-name
        self.Q1 = nn.Sequential(
            nn.Linear(self.q_input_size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ELU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.feature_layer_two = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
        )

        # Q2 architecture
        # pylint: disable-next=invalid-name
        self.Q2 = nn.Sequential(
            nn.Linear(self.q_input_size, hidden_sizes[0]),
            nn.ELU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ELU(),
            nn.Linear(hidden_sizes[1], 1),
        )


class Critic(BaseCritic):
    def __init__(self, observation_size: int, num_actions: int, config: TD7Config):

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            config=config,
        )
