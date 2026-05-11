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
        agent_networks: dict[str, MARLAlgorithm[dict[str, np.ndarray]]],
        env_teams: dict[str, list[str]],
        config: MultiMARLConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agent_networks

        self.env_teams = env_teams

        # Only the first agent is presumned to be learning, the rest are treated as part of the environment dynamics
        self.learning_team_name = config.learning_team_name
        self.learning_team = self.env_teams[self.learning_team_name]

        self.learning_agent_network = self.agent_networks[self.learning_team_name]
        self.learning_agent_team = self.env_teams[self.learning_team_name]

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_ids = list(observation.agent_states.keys())

        full_actions: dict[str, np.ndarray] = {}
        full_extras: dict[str, Any] = {}

        for agent_team_name, agent_network in self.agent_networks.items():
            action_sample = agent_network.act(
                observation=observation, evaluation=evaluation
            )

            team = self.env_teams[agent_team_name]
            for agent_name in team:
                full_actions[agent_name] = action_sample.action[agent_name]
                full_extras[agent_name] = action_sample.extras[agent_name]

        missing_agent_ids = set(agent_ids) - set(full_actions.keys())
        if missing_agent_ids:
            missing_keys = ", ".join(sorted(missing_agent_ids))
            raise KeyError(f"Missing actions for agents: {missing_keys}")

        return ActionSample(action=full_actions, extras=full_extras, source="policy")

    def train(
        self,
        memory_buffer: MARLMemoryBuffer,
        episode_context: EpisodeContext,
    ) -> dict[str, Any]:

        return self.learning_agent_network.train(
            memory_buffer=memory_buffer, episode_context=episode_context
        )

    def save_models(self, filepath: str, filename: str) -> None:
        if not os.path.exists(filepath):
            os.makedirs(filepath)

        for agent_name, agent_network in self.agent_networks.items():
            agent_filepath = os.path.join(filepath, f"{agent_name}")
            agent_filename = f"{filename}_{agent_name}_checkpoint"
            agent_network.save_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for agent_name, agent_network in self.agent_networks.items():
            agent_filepath = os.path.join(filepath, f"{agent_name}")
            agent_filename = f"{filename}_{agent_name}_checkpoint"
            agent_network.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
