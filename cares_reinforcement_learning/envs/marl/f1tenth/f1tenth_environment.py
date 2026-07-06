from functools import cached_property
from typing import Any

import numpy as np
from f1tenth_environments.environment_factory import EnvironmentFactory

from cares_reinforcement_learning.envs.configurations import F1TenthConfig
from cares_reinforcement_learning.envs.marl.marl_environment import MARLEnvironment
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation
from cares_reinforcement_learning.util import helpers as hlp


class F1TenthMARLEnvironment(MARLEnvironment):
    """Adapt the ROS F1Tenth parallel environment to the CARES-RL MARL API."""

    def __init__(self, config: F1TenthConfig, seed: int) -> None:
        super().__init__(config, seed)

        self.factory = EnvironmentFactory()
        self.env = self.factory.create("MultiCarRace", {})
        self.possible_agents = list(self.env.agents)
        self.agent_teams = self._split_agents_by_team(self.possible_agents)
        self.apply_action_normalization = True
        self.observation: MARLObservation

        self.set_seed(seed)

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
        environment_space = self.env.observation_size
        per_agent_space = dict(environment_space["obs"])
        return {
            "obs": per_agent_space,
            "state": int(environment_space["state"]),
            "num_agents": len(self.possible_agents),
            "teams": self.agent_teams,
        }

    @cached_property
    def num_agents(self) -> int:
        return len(self.possible_agents)

    @cached_property
    def action_num(self) -> int:
        return int(np.asarray(self.env.max_actions).shape[0])

    def get_available_actions(self) -> dict[str, np.ndarray]:
        return {
            agent: np.ones(self.action_num, dtype=np.int32)
            for agent in self.possible_agents
        }

    def sample_action(self) -> dict[str, np.ndarray]:
        actions = {}
        for agent in self.possible_agents:
            action = np.random.uniform(
                self.min_action_value[agent],
                self.max_action_value[agent],
            ).astype(np.float32)
            actions[agent] = hlp.normalize(
                action,
                self.max_action_value[agent],
                self.min_action_value[agent],
            )
        return actions

    def set_seed(self, seed: int) -> None:
        self.seed = seed
        self.env.set_seed(seed)

    def _build_observation(
        self, agent_states: dict[str, np.ndarray]
    ) -> MARLObservation:
        ordered_states = {
            agent: np.asarray(agent_states[agent], dtype=np.float32)
            for agent in self.possible_agents
        }
        global_state = np.concatenate(
            [ordered_states[agent].reshape(-1) for agent in self.possible_agents]
        )
        return MARLObservation(
            global_state=global_state,
            agent_states=ordered_states,
            available_actions=self.get_available_actions(),
        )

    def reset(self, training: bool = True) -> MARLObservation:
        agent_states, _ = self.env.reset(seed=self.seed)
        self.observation = self._build_observation(agent_states)
        return self.observation

    def step(self, action: dict[str, np.ndarray]) -> MultiAgentExperience:
        environment_action = {}
        for agent in self.possible_agents:
            environment_action[agent] = hlp.denormalize(
                np.asarray(action[agent], dtype=np.float32),
                self.max_action_value[agent],
                self.min_action_value[agent],
            )

        agent_states, rewards, dones, truncations, infos = self.env.step(
            environment_action
        )
        next_observation = self._build_observation(agent_states)

        experience = MultiAgentExperience(
            observation=self.observation.clone(),
            action={
                agent: np.asarray(action[agent], dtype=np.float32).copy()
                for agent in self.possible_agents
            },
            reward={agent: float(rewards[agent]) for agent in self.possible_agents},
            next_observation=next_observation.clone(),
            done={agent: bool(dones[agent]) for agent in self.possible_agents},
            truncated={
                agent: bool(truncations[agent]) for agent in self.possible_agents
            },
            info=infos,
        )

        self.observation = next_observation
        return experience

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_overlay_info(self) -> dict:
        return {}
