import torch
import torch.nn.functional as F
from torch import nn

from cares_reinforcement_learning.networks.common import MLP
from cares_reinforcement_learning.util.configurations import QMIXConfig


class BaseQMixer(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        network: nn.Module,
    ):
        super().__init__()

        self.observation_size = observation_size
        self.num_actions = num_actions
        self.network = network

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        output = self.network(state)
        return output


# This is the default base network for DQN for reference and testing of default network configurations
class DefaultQMixer(BaseQMixer):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
    ):
        hidden_sizes = [64, 64]

        network = nn.Sequential(
            nn.Linear(observation_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )
        super().__init__(
            observation_size=observation_size, num_actions=num_actions, network=network
        )


# class QMixer(BaseQMixer):
#     def __init__(self, observation_size: int, num_actions: int, config: DQNConfig):

#         network = MLP(
#             input_size=observation_size,
#             output_size=num_actions,
#             config=config.network_config,
#         )
#         super().__init__(
#             observation_size=observation_size, num_actions=num_actions, network=network
#         )


class QMixer(nn.Module):
    def __init__(
        self, num_agents: int, state_dim: int, embed_dim: int = 32, hidden_dim: int = 64
    ):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim  # hypernetwork embedding size

        # Hypernetwork for the first layer weights
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, num_agents * embed_dim),
        )
        # Hypernetwork for first layer bias
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernetwork for second layer weights (output layer)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
        )
        # Hypernetwork for second layer bias
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),  # scalar output bias
        )

    def forward(self, agent_qs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        agent_qs: [batch_size, n_agents]
        state: [batch_size, state_dim]
        """
        batch_size = agent_qs.size(0)

        # First layer
        # [B, n_agents, embed_dim]
        w1 = self.hyper_w1(state).view(batch_size, self.num_agents, self.embed_dim)
        w1 = torch.abs(w1)  # enforce non-negative weights

        # [B, 1, embed_dim]
        b1 = self.hyper_b1(state).view(batch_size, 1, self.embed_dim)

        # Compute hidden layer
        # [B, 1, embed_dim]
        hidden = torch.bmm(agent_qs.unsqueeze(1), w1) + b1
        hidden = F.relu(hidden)

        # Second layer (output)
        # [B, embed_dim, 1]
        w2 = self.hyper_w2(state).view(batch_size, self.embed_dim, 1)
        w2 = torch.abs(w2)  # enforce non-negative weights

        # [B, 1, 1]
        b2 = self.hyper_b2(state).view(batch_size, 1, 1)

        # [B, 1, 1]
        q_tot = torch.bmm(hidden, w2) + b2

        # [batch_size]
        return q_tot.view(-1)
