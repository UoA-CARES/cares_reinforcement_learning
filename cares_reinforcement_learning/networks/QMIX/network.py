"""
This is a stub file for the DQN network - reads directly off DQN's Network class.
"""

import torch
import torch.nn as nn

from cares_reinforcement_learning.networks.DQN import Network
from cares_reinforcement_learning.util.configurations import DQNConfig, QMIXConfig


class MultiAgentNetwork(nn.Module):
    def __init__(
        self,
        observation_size: int,
        num_actions: int,
        num_agents: int,
        config: DQNConfig,
    ):
        super().__init__()
        self.observation_size = observation_size
        self.num_actions = num_actions

        self.ensemble_size = num_agents

        self.agents: list[Network | nn.Sequential] = []

        for i in range(self.ensemble_size):
            agent_net = Network(
                observation_size=self.observation_size,
                num_actions=self.num_actions,
                config=config,
            )
            self.add_module(f"agent_net_{i}", agent_net)
            self.agents.append(agent_net)

    def forward(
        self, observations: torch.Tensor, actions: torch.Tensor
    ) -> torch.Tensor:
        """
        observations: [batch_size, num_agents, obs_dim]
        actions: [batch_size, num_agents]
        returns: agent_qs [batch_size, num_agents]
        """

        agent_qs_list: list[torch.Tensor] = []

        for i, agent in enumerate(self.agents):
            # Each agent observes only its local state
            obs_i = observations[:, i, :]  # [batch, obs_dim]
            q_values_i = agent(obs_i)  # [batch, num_actions]

            # Select the Q-value corresponding to the chosen action
            action_i = actions[:, i].unsqueeze(1)  # [batch, 1]
            q_i = q_values_i.gather(1, action_i).squeeze(1)  # [batch]
            agent_qs_list.append(q_i)

        # Stack each agent’s selected Q-value → [batch, num_agents]
        agent_qs_tensor = torch.stack(agent_qs_list, dim=1)
        return agent_qs_tensor
