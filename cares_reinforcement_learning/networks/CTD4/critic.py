from torch import nn

from cares_reinforcement_learning.networks.common import (
    BaseCritic,
    ContinuousDistributedCritic,
    EnsembleCritic,
)
from cares_reinforcement_learning.util.configurations import CTD4Config


# This is the default base network for CTD4 for reference and testing of default network configurations
class DefaultContinuousDistributedCritic(ContinuousDistributedCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, action_num: int):
        input_size = observation_size + action_num
        hidden_sizes = [256, 256]

        # pylint: disable-next=non-parent-init-called
        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=1,
        )

        self.mean_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.std_layer = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], 1),
        )

        self.soft_std_layer = nn.Softplus()


class DefaultCritic(EnsembleCritic):
    # pylint: disable=super-init-not-called
    def __init__(self, observation_size: int, num_actions: int):
        input_size = observation_size + num_actions

        ensemble_size = 3

        # pylint: disable-next=non-parent-init-called
        BaseCritic.__init__(
            self,
            input_size=input_size,
            output_size=1,
        )

        self.critics: list[BaseCritic | nn.Sequential] = []

        for i in range(ensemble_size):
            critic_net = DefaultContinuousDistributedCritic(
                observation_size=observation_size, action_num=num_actions
            )
            self.add_module(f"critic_net_{i}", critic_net)
            self.critics.append(critic_net)


class Critic(EnsembleCritic):
    def __init__(self, observation_size: int, num_actions: int, config: CTD4Config):
        input_size = observation_size + num_actions

        super().__init__(
            input_size=input_size,
            output_size=1,
            ensemble_size=config.ensemble_size,
            config=config.critic_config,
            critic_type=ContinuousDistributedCritic,
        )
