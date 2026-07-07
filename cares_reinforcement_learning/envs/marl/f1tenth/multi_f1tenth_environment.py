from functools import cached_property
from typing import Any

import numpy as np

from f1tenth_environments.environment_factory import EnvironmentFactory
from cares_reinforcement_learning.envs.marl.marl_environment import MARLEnvironment
from cares_reinforcement_learning.envs.configurations import MultiF1TenthConfig
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation
from cares_reinforcement_learning.util import helpers as hlp


class F1TenthMARLEnvironment(MARLEnvironment):
    def __init__(self, config: MultiF1TenthConfig, seed: int) -> None:
        super().__init__(config, seed)

        self.factory = EnvironmentFactory()
        self.env = self.factory.create(config.task, {})

        # Stable agent ordering: all list/dict outputs follow this order.
        self.possible_agents: list[str] = list(self.env.agents)
        self.agent_teams = self._split_agents_by_team(self.possible_agents)
        self.apply_action_normalization = True
        self.observation: MARLObservation

        self.set_seed(seed)

    # ------------------------------------------------------------------
    # MARLEnvironment abstract properties
    # ------------------------------------------------------------------

    @cached_property
    def max_action_value(self) -> dict[str, np.ndarray]:
        return {
            agent: np.asarray(self.env.max_actions, dtype=np.float32).copy()
            for agent in self.possible_agents
        }

    @cached_property
    def min_action_value(self) -> dict[str, np.ndarray]:
        return {
            agent: np.asarray(self.env.min_actions, dtype=np.float32).copy()
            for agent in self.possible_agents
        }

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        raw = self.env.observation_size
        return {
            "obs": dict(raw["obs"]),  # dict[agent_id -> per-agent obs dim]
            "state": int(raw["state"]),  # int, concatenated global state dim
            "num_agents": int(raw["num_agents"]),
            "teams": dict(raw.get("teams", self.agent_teams)),
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
        self.seed = seed
        if hasattr(self.env, "set_seed"):
            self.env.set_seed(seed)

        for index, agent in enumerate(self.possible_agents):
            action_space = getattr(self.env, "action_spaces", {}).get(agent)
            if hasattr(action_space, "seed"):
                action_space.seed(seed + index)

    def sample_action(self) -> dict[str, np.ndarray]:
        actions = {}
        for agent in self.possible_agents:
            action = self.env.action_spaces[agent].sample()
            if self.apply_action_normalization:
                action = hlp.normalize(
                    action,
                    self.max_action_value[agent],
                    self.min_action_value[agent],
                )
            actions[agent] = np.asarray(action, dtype=np.float32)
        return actions

    def reset(self, training: bool = True) -> MARLObservation:
        obs_dict, _ = self.env.reset()

        self.observation = MARLObservation(
            global_state=self._build_global_state(obs_dict),
            agent_states=obs_dict,
            available_actions=self.get_available_actions(),
        )

        return self.observation

    def step(self, action: dict[str, np.ndarray]) -> MultiAgentExperience:
        environment_action = action.copy()
        if self.apply_action_normalization:
            environment_action = {
                agent: hlp.denormalize(
                    np.asarray(action[agent], dtype=np.float32),
                    self.max_action_value[agent],
                    self.min_action_value[agent],
                )
                for agent in self.possible_agents
            }

        obs_dict, rewards, terminateds, truncateds, infos = self.env.step(
            environment_action
        )

        next_observation = MARLObservation(
            global_state=self._build_global_state(obs_dict),
            agent_states=obs_dict,
            available_actions=self.get_available_actions(),
        )

        experience = MultiAgentExperience(
            observation=self.observation.clone(),
            action={
                agent: np.asarray(action[agent], dtype=np.float32).copy()
                for agent in self.possible_agents
            },
            reward={a: rewards[a] for a in self.possible_agents},
            next_observation=next_observation.clone(),
            done={a: terminateds[a] for a in self.possible_agents},
            truncated={a: truncateds[a] for a in self.possible_agents},
            info=infos,
        )

        self.observation = next_observation
        return experience
