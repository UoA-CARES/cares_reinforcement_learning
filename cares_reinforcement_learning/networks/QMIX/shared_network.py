"""
Implements multi-agent network architectures for QMIX, including independent and shared multi-agent networks.
These classes use the DQN Network as a component for each agent or as a shared network.
"""

import torch
import torch.nn as nn

from cares_reinforcement_learning.networks.DQN import Network
from cares_reinforcement_learning.util.configurations import QMIXConfig


class BaseSharedMultiAgentNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: int,
        num_actions: int,
        num_agents: int,
        agent: nn.Module,
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        self.agent = agent

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [batch_size, num_agents, obs_dim]
        returns: [batch_size, num_agents, num_actions]
        """

        batch_size, num_agents, obs_dim = observations.shape

        # Flatten for shared forward pass
        obs_flat = observations.reshape(batch_size * num_agents, obs_dim)

        # Shared forward pass
        q_values = self.agent(obs_flat)  # [B*N, num_actions]

        # Reshape back
        q_values = q_values.view(batch_size, num_agents, self.num_actions)
        return q_values


class DefaultSharedMultiAgentNetwork(BaseSharedMultiAgentNetwork):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
    ):
        # Shared network for all agents
        # Note: add agent ID embedding dimension (num_agents)
        obs_shape = observation_size["obs"]
        num_agents = observation_size["num_agents"]
        hidden_sizes = [64, 64]

        input_size = obs_shape
        agent = nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], num_actions),
        )

        super().__init__(
            obs_shape=obs_shape,
            num_actions=num_actions,
            num_agents=num_agents,
            agent=agent,
        )


class SharedMultiAgentNetwork(BaseSharedMultiAgentNetwork):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
        config: QMIXConfig,
    ):
        # Shared network for all agents
        # Note: add agent ID embedding dimension (num_agents)
        obs_shape = observation_size["obs"]
        num_agents = observation_size["num_agents"]

        input_size = obs_shape
        agent = Network(
            observation_size=input_size,
            num_actions=num_actions,
            config=config,
        )
        super().__init__(
            obs_shape=obs_shape,
            num_actions=num_actions,
            num_agents=num_agents,
            agent=agent,
        )
