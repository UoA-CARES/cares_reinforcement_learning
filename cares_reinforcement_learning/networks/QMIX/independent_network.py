"""
Independent multi-agent network for QMIX.
Each agent has its own network based on DQN's Network class.
"""

import torch
import torch.nn as nn

from cares_reinforcement_learning.networks.DQN import Network
from cares_reinforcement_learning.util.configurations import QMIXConfig


class BaseIndependentMultiAgentNetwork(nn.Module):
    def __init__(
        self,
        obs_shape: int,
        num_actions: int,
        num_agents: int,
        agents: list[Network | nn.Sequential],
    ):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_actions = num_actions
        self.num_agents = num_agents

        self.agents: list[Network | nn.Sequential] = agents

        for i, agent in enumerate(self.agents):
            self.add_module(f"agent_net_{i}", agent)

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


class DefaultIndependentMultiAgentNetwork(BaseIndependentMultiAgentNetwork):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
    ):
        agents: list[Network | nn.Sequential] = []

        obs_shape = observation_size["obs"]
        num_agents = observation_size["num_agents"]

        for _ in range(num_agents):
            agent_net = nn.Sequential(
                nn.Linear(obs_shape, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, num_actions),
            )
            agents.append(agent_net)

        super().__init__(
            obs_shape=obs_shape,
            num_actions=num_actions,
            num_agents=num_agents,
            agents=agents,
        )


class IndependentMultiAgentNetwork(BaseIndependentMultiAgentNetwork):
    def __init__(
        self,
        observation_size: dict,
        num_actions: int,
        config: QMIXConfig,
    ):
        agents: list[Network | nn.Sequential] = []

        obs_shape = observation_size["obs"]
        num_agents = observation_size["num_agents"]

        for _ in range(num_agents):
            agent_net = Network(
                observation_size=obs_shape,
                num_actions=num_actions,
                config=config,
            )
            agents.append(agent_net)

        super().__init__(
            obs_shape=obs_shape,
            num_actions=num_actions,
            num_agents=num_agents,
            agents=agents,
        )
