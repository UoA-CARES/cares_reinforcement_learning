"""
This is a stub file for the DQN network - reads directly off DQN's Network class.
"""

import torch
import torch.nn as nn

from cares_reinforcement_learning.networks.DQN import Network
from cares_reinforcement_learning.util.configurations import QMIXConfig


class IndependentMultiAgentNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_agents: int,
        config: QMIXConfig,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.num_agents = num_agents

        self.agents: list[Network | nn.Sequential] = []

        for i in range(self.num_agents):
            agent_net = Network(
                observation_size=self.observation_size,
                num_actions=self.num_actions,
                config=config,
            )
            self.add_module(f"agent_net_{i}", agent_net)
            self.agents.append(agent_net)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        observations: [batch, num_agents, obs_dim]
        returns: [batch, num_agents, num_actions]
        """

        agent_qs_list: list[torch.Tensor] = []

        for i, agent in enumerate(self.agents):
            # Each agent observes only its local state
            obs_i = observations[:, i, :]  # [batch, obs_dim]
            q_values_i = agent(obs_i)  # [batch, num_actions]

            agent_qs_list.append(q_values_i)

        agent_qs_tensor = torch.stack(agent_qs_list, dim=1)  # [B, N, num_actions]
        return agent_qs_tensor


class SharedMultiAgentNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_agents: int,
        config: QMIXConfig,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions
        self.num_agents = num_agents

        # Shared network for all agents
        # Note: add agent ID embedding dimension (num_agents)
        self.agent = Network(
            observation_size=self.observation_size + self.num_agents,
            num_actions=self.num_actions,
            config=config,
        )

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
