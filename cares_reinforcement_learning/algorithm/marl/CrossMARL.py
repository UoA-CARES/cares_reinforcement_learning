import logging
import os
from typing import Any

import numpy as np
import torch

from cares_reinforcement_learning.algorithm.algorithm import MARLAlgorithm
from cares_reinforcement_learning.algorithm.configurations import CrossMARLConfig
from cares_reinforcement_learning.memory.memory_buffer import MARLMemoryBuffer
from cares_reinforcement_learning.types.action import ActionSample
from cares_reinforcement_learning.types.episode import EpisodeContext
from cares_reinforcement_learning.types.observation import MARLObservation


class CrossMARL(MARLAlgorithm[dict[str, np.ndarray]]):
    config: CrossMARLConfig

    def __init__(
        self,
        agent_networks: dict[str, MARLAlgorithm[dict[str, np.ndarray]]],
        env_teams: dict[str, list[str]],
        config: CrossMARLConfig,
        device: torch.device,
    ):
        super().__init__(policy_type="policy", config=config, device=device)

        self.agent_networks = agent_networks

        self.env_teams = env_teams

        # Only the first agent is presumned to be learning, the rest are treated as part of the environment dynamics
        self.learning_team_name = config.learning_team_name
        self.learning_team = (
            self.env_teams[self.learning_team_name]
            if self.learning_team_name is not None
            else []
        )
        self.learning_agent_network = (
            self.agent_networks[self.learning_team_name]
            if self.learning_team_name is not None
            else None
        )

        if self.learning_team_name is not None:
            self._load_frozen_models()

    def _load_frozen_models(self) -> None:
        for agent_name, agent_network in self.agent_networks.items():
            if agent_name == self.learning_team_name:
                continue

            agent_config = self.config.agents_config[agent_name]
            model_path = getattr(agent_config, "model_path", None)

            if model_path is None:
                raise ValueError(
                    f"Frozen CrossMARL team '{agent_name}' is missing model_path."
                )

            agent_network.load_models(model_path, agent_config.algorithm)

    def _merge_team_extras(
        self,
        full_extras: dict[str, Any],
        team: list[str],
        action_extras: dict[str, Any] | None,
    ) -> None:
        if not action_extras:
            return

        for key, value in action_extras.items():
            if isinstance(value, dict):
                full_extras.setdefault(key, {})

                for agent_name in team:
                    if agent_name in value:
                        full_extras[key][agent_name] = value[agent_name]
            else:
                full_extras[key] = value

    def act(
        self,
        observation: MARLObservation,
        evaluation: bool = False,
    ) -> ActionSample[dict[str, np.ndarray]]:
        agent_ids = list(observation.agent_states.keys())

        full_actions: dict[str, np.ndarray] = {}
        full_extras: dict[str, Any] = {}

        for agent_team_name, agent_network in self.agent_networks.items():
            team_evaluation = evaluation or agent_team_name != self.learning_team_name
            action_sample = agent_network.act(
                observation=observation, evaluation=team_evaluation
            )

            team = self.env_teams[agent_team_name]
            for agent_name in team:
                full_actions[agent_name] = action_sample.action[agent_name]

            self._merge_team_extras(
                full_extras=full_extras,
                team=team,
                action_extras=action_sample.extras,
            )

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
        if self.learning_agent_network is None:
            raise ValueError("No learning team specified for CrossMARL setup.")

        return self.learning_agent_network.train(
            memory_buffer=memory_buffer, episode_context=episode_context
        )

    def _get_agent_model_path(self, filepath: str, agent_name: str) -> str:
        return os.path.join(filepath, agent_name)

    def _get_agent_model_filename(self, agent_name: str) -> str:
        agent_config = self.config.agents_config[agent_name]
        return agent_config.algorithm

    def save_models(self, filepath: str, filename: str) -> None:
        if self.learning_agent_network is None or self.learning_team_name is None:
            logging.info("CrossMARL has no learning team; skipping model save.")
            return

        os.makedirs(filepath, exist_ok=True)

        agent_filepath = os.path.join(filepath, self.learning_team_name)
        agent_filename = f"{filename}_{self.learning_team_name}_checkpoint"

        self.learning_agent_network.save_models(agent_filepath, agent_filename)

        logging.info("learning team models and optimisers have been saved...")

    def load_models(self, filepath: str, filename: str) -> None:
        for agent_name, agent_network in self.agent_networks.items():
            # Learning team is fresh during train-against-fixed.
            if agent_name == self.learning_team_name:
                continue

            agent_filepath = self._get_agent_model_path(filepath, agent_name)
            agent_filename = self._get_agent_model_filename(agent_name)

            agent_network.load_models(agent_filepath, agent_filename)

        logging.info("models and optimisers have been loaded...")
