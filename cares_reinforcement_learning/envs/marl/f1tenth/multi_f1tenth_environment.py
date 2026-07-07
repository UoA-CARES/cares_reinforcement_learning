from functools import cached_property
from typing import Any

import numpy as np

from f1tenth_environments.environment_factory import EnvironmentFactory
from cares_reinforcement_learning.envs.marl.marl_environment import MARLEnvironment
from cares_reinforcement_learning.envs.configurations import MultiF1TenthConfig
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation


class F1TenthMARLEnvironment(MARLEnvironment):
    def __init__(self, config: MultiF1TenthConfig, seed: int) -> None:
        super().__init__(config, seed)

        # Import here to avoid a hard ROS2 dependency at module load time
        self.factory = EnvironmentFactory()

        # # Instantiate the task
        self.env = self.factory.create(config.task, {})

        # Stable agent ordering — all list outputs follow this order
        self.possible_agents: list[str] = self.env.agents

        self.observation: MARLObservation

<<<<<<< HEAD
=======
        self.agent_teams = self._split_agents_by_team(self.possible_agents)

>>>>>>> main
    # ------------------------------------------------------------------
    # MARLEnvironment abstract properties
    # ------------------------------------------------------------------

    @cached_property
    def max_action_value(self) -> list[np.ndarray]:
        return [self.env.max_actions.copy() for _ in self.possible_agents]

    @cached_property
    def min_action_value(self) -> list[np.ndarray]:
        return [self.env.min_actions.copy() for _ in self.possible_agents]

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        raw = self.env.observation_size
        return {
            "obs": dict(raw["obs"]),  # dict[agent_id -> per-agent obs dim]
            "state": int(raw["state"]),  # int, concatenated global state dim
            "num_agents": int(raw["num_agents"]),
            "teams": dict(raw["teams"]),
        }

    @cached_property
    def action_num(self) -> int:
        return int(self.env.max_actions.shape[0])

    @cached_property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_global_state(self, obs_dict: dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            [obs_dict[agent] for agent in self.possible_agents], axis=0
        )

    def get_available_actions(self) -> dict[str, np.ndarray]:
        return {
            agent: np.ones(self.action_num, dtype=np.int32)
            for agent in self.possible_agents
        }

    def get_overlay_info(self) -> dict:
        return {}

    # ------------------------------------------------------------------
    # MARLEnvironment abstract methods
    # ------------------------------------------------------------------

    def set_seed(self, seed: int) -> None:
        # F1Tenth runs in ROS2/Gazebo — seeding is a no-op, kept for
        # interface compliance
        self.seed = seed

    def sample_action(self) -> dict[str, np.ndarray]:
        return {
            agent: self.env.action_spaces[agent].sample()
            for agent in self.possible_agents
        }

    def reset(self, training: bool = True) -> MARLObservation:
        obs_dict, _ = self.env.reset()

        self.observation = MARLObservation(
            global_state=self._build_global_state(obs_dict),
            agent_states=obs_dict,
            available_actions=self.get_available_actions(),
        )

        return self.observation

    def step(self, action: dict[str, np.ndarray]) -> MultiAgentExperience:
        obs_dict, rewards, terminateds, truncateds, infos = self.env.step(action)

        next_observation = MARLObservation(
            global_state=self._build_global_state(obs_dict),
            agent_states=obs_dict,
            available_actions=self.get_available_actions(),
        )

        experience = MultiAgentExperience(
            observation=self.observation.clone(),
            action=action,
            reward={a: rewards[a] for a in self.possible_agents},
            next_observation=next_observation.clone(),
            done={a: terminateds[a] for a in self.possible_agents},
            truncated={a: truncateds[a] for a in self.possible_agents},
            info=infos,
        )

        self.observation = next_observation
        return experience
