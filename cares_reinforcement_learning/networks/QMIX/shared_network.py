"""
This is a stub file for the DQN network - reads directly off DQN's Network class.
"""

import torch
import torch.nn as nn

from cares_reinforcement_learning.networks.DQN import Network
from cares_reinforcement_learning.util.configurations import QMIXConfig


class BaseSharedMultiAgentNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_agents: int,
        agent: nn.Module,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.num_agents = num_agents

        self.agent = agent

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [batch_size, num_agents, obs_dim]
        returns: [batch_size, num_agents, num_actions]
        """
        batch_size, num_agents, obs_dim = observations.shape
        device = observations.device

        # Create one-hot agent IDs
        agent_ids = torch.eye(self.num_agents, device=device)
        agent_ids = agent_ids.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N, N]

        # Concatenate IDs to observations
        obs_with_id = torch.cat([observations, agent_ids], dim=-1)  # [B, N, obs_dim+N]

        # Flatten for shared forward pass
        obs_with_id = obs_with_id.reshape(batch_size * num_agents, -1)

        # Shared forward pass
        q_values = self.agent(obs_with_id)  # [B*N, num_actions]

        # Reshape back
        q_values = q_values.view(batch_size, num_agents, self.num_actions)
        return q_values


class DefaultSharedMultiAgentNetwork(BaseSharedMultiAgentNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_agents: int,
    ):
        # Shared network for all agents
        # Note: add agent ID embedding dimension (num_agents)
        hidden_sizes = [64, 64]

        input_size = observation_size + num_agents
        agent = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            num_agents=num_agents,
            agent=agent,
        )


class SharedMultiAgentNetwork(BaseSharedMultiAgentNetwork):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_agents: int,
        config: QMIXConfig,
    ):
        # Shared network for all agents
        # Note: add agent ID embedding dimension (num_agents)
        input_size = observation_size + num_agents
        agent = Network(
            observation_size=input_size,
            num_actions=num_actions,
            config=config,
        )
        super().__init__(
            observation_size=observation_size,
            num_actions=num_actions,
            num_agents=num_agents,
            agent=agent,
        )
