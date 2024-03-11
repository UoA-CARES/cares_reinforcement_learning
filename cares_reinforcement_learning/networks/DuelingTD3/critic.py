import logging

import numpy as np
import torch
from torch import nn


class Critic(nn.Module):
    def __init__(
        self, agent_observation_size, envrionment_observation_size, num_actions
    ):
        super().__init__()

        self.hidden_size = [128, 128]

        self.agent_observation_size = agent_observation_size
        self.envrionment_observation_size = envrionment_observation_size

        # Agent Q1 architecture
        # pylint: disable-next=invalid-name
        self.agent_Q1 = nn.Sequential(
            nn.Linear(agent_observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # Agent Q2 architecture
        # pylint: disable-next=invalid-name
        self.agent_Q2 = nn.Sequential(
            nn.Linear(agent_observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # Environment Q1 architecture
        # pylint: disable-next=invalid-name
        self.environment_Q1 = nn.Sequential(
            nn.Linear(envrionment_observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

        # Environment Q2 architecture
        # pylint: disable-next=invalid-name
        self.environment_Q2 = nn.Sequential(
            nn.Linear(envrionment_observation_size + num_actions, self.hidden_size[0]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[0], self.hidden_size[1]),
            nn.ReLU(),
            nn.Linear(self.hidden_size[1], 1),
        )

    def _split_state(self, state):
        def extract(lst, start1, end1, start2, end2):
            return [item[start1:end1] + item[start2:end2] for item in lst]

        agent_state = extract(state.tolist(), 0, 2, 4, 6)

        environment_sate = extract(state.tolist(), 2, 4, 6, 8)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        agent_state = torch.FloatTensor(np.asarray(agent_state)).to(device)
        environment_sate = torch.FloatTensor(np.asarray(environment_sate)).to(device)
        return agent_state, environment_sate

    def forward(self, state, action):
        agent_state, environment_sate = self._split_state(state)

        obs_agent = torch.cat([agent_state, action], dim=1)
        obs_envrionment = torch.cat([environment_sate, action], dim=1)

        agent_q1 = self.agent_Q1(obs_agent)
        environment_q1 = self.environment_Q1(obs_envrionment)

        q1 = agent_q1 + environment_q1

        agent_q2 = self.agent_Q2(obs_agent)
        environment_q2 = self.environment_Q2(obs_envrionment)

        q2 = agent_q2 + environment_q2

        return q1, q2
