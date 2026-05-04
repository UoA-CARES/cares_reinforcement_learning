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


class MultiMARL(MARLAlgorithm[list[np.ndarray]]):
    def __init__(
        self,
        agent_networks: list[MARLAlgorithm[list[np.ndarray]]],
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
        agent_ids = list(global_observation.agent_states.keys())
        agent_index = {agent_id: i for i, agent_id in enumerate(agent_ids)}

        team_indices = [agent_index[agent_id] for agent_id in team_agents]

        return MARLObservation(
            agent_states={
                agent_id: global_observation.agent_states[agent_id]
                for agent_id in team_agents
            },
            global_state=global_observation.global_state,
            avail_actions=global_observation.avail_actions[team_indices],
        )

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[list[np.ndarray]]:
        agent_ids = list(observation.agent_states.keys())
        agent_index = {agent_id: i for i, agent_id in enumerate(agent_ids)}

        full_actions: list[np.ndarray | None] = [None] * len(agent_ids)

        # Learning agent only sees/acts for its team
        learning_observation = self._observation_for_team(
            team_agents=self.learning_agent_team,
            global_observation=observation,
        )

        learning_action_sample = self.learning_agent.act(
            observation=learning_observation,
            evaluation=evaluation,
        )

        for local_index, agent_id in enumerate(self.learning_agent_team):
            global_index = agent_index[agent_id]
            full_actions[global_index] = learning_action_sample.action[local_index]

        # Pretrained/full-task algorithms see full observation,
        # but we only use actions for their assigned team.
        for agent, team_agents in zip(self.agent_networks[1:], self.agents_team[1:]):
            action_sample = agent.act(
                observation=observation,
                evaluation=evaluation,
            )

            for agent_id in team_agents:
                global_index = agent_index[agent_id]
                full_actions[global_index] = action_sample.action[global_index]

        if any(action is None for action in full_actions):
            missing_agents = [
                agent_id
                for agent_id, action in zip(agent_ids, full_actions)
                if action is None
            ]
            raise ValueError(f"No action provided for agents: {missing_agents}")

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
