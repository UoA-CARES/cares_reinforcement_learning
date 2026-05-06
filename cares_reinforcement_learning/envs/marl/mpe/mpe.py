from functools import cached_property
from typing import Any

import cv2
import numpy as np
from gymnasium import spaces
from mpe2 import all_modules as mpe_all
from pettingzoo.utils.env import AgentID, ParallelEnv

from cares_reinforcement_learning.envs.marl.marl_environment import (
    MARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import MPEConfig
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation
from cares_reinforcement_learning.util import helpers as hlp

ALL_ENV_MODULES = {
    **mpe_all.mpe_environments,
}


def make_env(env_name: str, render_mode=None, continuous_actions=False) -> ParallelEnv:
    if f"mpe/{env_name}" not in ALL_ENV_MODULES:
        raise ValueError(
            f"Unknown environment '{env_name}'. Available: {list(ALL_ENV_MODULES.keys())}"
        )

    module = ALL_ENV_MODULES[f"mpe/{env_name}"]
    return (module.parallel_env)(
        render_mode=render_mode, continuous_actions=continuous_actions
    )


class MPE2Environment(MARLEnvironment):
    def __init__(self, config: MPEConfig, seed: int) -> None:
        super().__init__(config, seed)

        self.continuous_actions = bool(config.continuous_actions)

        self.env = make_env(
            env_name=self.task,
            render_mode="rgb_array",
            continuous_actions=self.continuous_actions,
        )

        self.possible_agents: list[AgentID] = self.env.possible_agents

        self.observation: MARLObservation

        self.set_seed(self.seed)

        self.apply_action_normalization = self.continuous_actions

    @cached_property
    def max_action_value(self) -> dict[str, np.ndarray]:
        max_action_values = {}
        for agent in self.env.agents:
            if isinstance(self.env.action_space(agent), spaces.Box):
                max_action_values[agent] = self.env.action_space(agent).high
            else:
                raise ValueError("Action space is not continuous")
        return max_action_values

    @cached_property
    def min_action_value(self) -> dict[str, np.ndarray]:
        min_action_values = {}
        for agent in self.env.agents:
            if isinstance(self.env.action_space(agent), spaces.Box):
                min_action_values[agent] = self.env.action_space(agent).low
            else:
                raise ValueError("Action space is not continuous")
        return min_action_values

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        """
        Return observation and state dimensions for each agent and the global critic.
        """

        # 1. Per-agent observation shapes (may differ!)
        obs_spaces = {
            agent: self.env.observation_space(agent).shape[0]
            for agent in self.env.agents
        }

        # 2. Global state shape (single vector from env.state())
        state_shape = self.env.state_space.shape[0]

        # 3. Number of agents
        num_agents = self.env.max_num_agents

        return {
            "obs": obs_spaces,  # dict[str → obs_dim_i]
            "state": state_shape,  # scalar int
            "num_agents": num_agents,  # int
        }

    @cached_property
    def num_agents(self) -> int:
        return self.env.num_agents

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.env.action_space(self.env.agents[0]), spaces.Box):
            action_num = self.env.action_space(self.env.agents[0]).shape[0]
        elif isinstance(self.env.action_space(self.env.agents[0]), spaces.Discrete):
            action_num = self.env.action_space(self.env.agents[0]).n
        else:
            raise ValueError(
                f"Unhandled action space type: {type(self.env.action_space)}"
            )
        return action_num

    def get_available_actions(self) -> dict[str, np.ndarray]:
        return {
            agent: np.ones(self.action_num, dtype=np.int32)
            for agent in self.possible_agents
        }

    def sample_action(self) -> dict[str, np.ndarray]:
        actions = {}
        for agent in self.possible_agents:
            space = self.env.action_space(agent)
            action = space.sample()

            if self.apply_action_normalization:
                action = hlp.normalize(
                    action,
                    self.max_action_value[agent],
                    self.min_action_value[agent],
                )

            actions[agent] = action

        return actions

    def set_seed(self, seed: int) -> None:
        self.seed = seed

        self.env.reset(seed=self.seed)

        # Seed action and observation spaces - different action seed per agent to avoid produciong the same values
        for i, agent in enumerate(self.env.agents):
            self.env.action_space(agent).seed(self.seed + i)
            self.env.observation_space(agent).seed(self.seed)

    def reset(self, training: bool = True) -> MARLObservation:
        """Reset PettingZoo parallel env and return MARL-compatible state dict."""
        agent_states, _ = self.env.reset()

        self.observation = MARLObservation(
            global_state=self.env.state(),
            agent_states=agent_states,
            available_actions=self.get_available_actions(),
        )

        return self.observation

    def step(self, action: dict[str, np.ndarray]) -> MultiAgentExperience:
        envrionment_action = action.copy()
        if self.apply_action_normalization:
            for agent in action.keys():
                envrionment_action[agent] = hlp.denormalize(
                    action[agent],
                    self.max_action_value[agent],
                    self.min_action_value[agent],
                )

        agent_states, rewards, dones, truncations, infos = self.env.step(
            envrionment_action
        )

        next_observation = MARLObservation(
            global_state=self.env.state(),
            agent_states=agent_states,
            available_actions=self.get_available_actions(),
        )

        experience = MultiAgentExperience(
            observation=self.observation.clone(),
            action=action,
            reward=rewards,
            next_observation=next_observation.clone(),
            done=dones,
            truncated=truncations,
            info=infos,
        )

        self.observation = next_observation

        return experience

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
