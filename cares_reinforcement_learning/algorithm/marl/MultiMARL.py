import logging
import os
from typing import Any

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import MultiMARLConfig
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import MARLObservation


class MultiMARL(MARLAlgorithm[dict[str, np.ndarray]]):
    def __init__(
        self,
        agent_networks: list[MARLAlgorithm[dict[str, np.ndarray]]],
        teams: dict[str, list[str]],
        config: MultiMARLConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agent_networks

        self.teams = teams

        # str list of names for each agent's team, e.g. ["adversary", "ally"]
        self.agents_team = config.agents_team

        # Only the first agent is presumned to be learning, the rest are treated as part of the environment dynamics
        self.learning_agent = self.agent_networks[0]
        self.learning_agent_team = self.teams[self.agents_team[0]]

    def _observation_for_team(
        self,
        team_agents: list[str],
        global_observation: MARLObservation,
    ) -> MARLObservation:

        agent_states = {}
        available_actions = {}
        for agent_id in team_agents:
            agent_states[agent_id] = global_observation.agent_states[agent_id]
            available_actions[agent_id] = global_observation.available_actions[agent_id]

        return MARLObservation(
            agent_states=agent_states,
            global_state=global_observation.global_state,
            available_actions=available_actions,
        )

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_ids = list(observation.agent_states.keys())

        full_actions: dict[str, np.ndarray] = {}

        # Learning agent only sees/acts for its team
        learning_observation = self._observation_for_team(
            team_agents=self.learning_agent_team,
            global_observation=observation,
        )

        learning_action_sample = self.learning_agent.act(
            observation=learning_observation,
            evaluation=evaluation,
        )

        for agent_id in self.learning_agent_team:
            full_actions[agent_id] = learning_action_sample.action[agent_id]

        # Pretrained/full-task algorithms see full observation,
        # but we only use actions for their assigned team.
        for agent, team_agents in zip(self.agent_networks[1:], self.agents_team[1:]):
            action_sample = agent.act(
                observation=observation,
                evaluation=evaluation,
            )

            for agent_id in team_agents:
                full_actions[agent_id] = action_sample.action[agent_id]

        missing_agent_ids = set(agent_ids) - set(full_actions.keys())
        if missing_agent_ids:
            missing_keys = ", ".join(sorted(missing_agent_ids))
            raise KeyError(f"Missing actions for agents: {missing_keys}")

        return ActionSample(action=full_actions, source="policy")

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        return {}

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for i, agent in enumerate(self.agent_networks):
            agent_filepath = os.path.join(filepath, f"agent_{i}")
            agent_filename = f"{filename}_agent_{i}_checkpoint"
            agent.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
