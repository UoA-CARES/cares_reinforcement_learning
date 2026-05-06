from functools import cached_property
from typing import Any

import cv2
import numpy as np
from smac.env import StarCraft2Env

from cares_reinforcement_learning.envs.configurations import SMACConfig
from cares_reinforcement_learning.envs.marl.marl_environment import (
    MARLEnvironment,
)
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation


class SMACEnvironment(MARLEnvironment):
    def __init__(self, config: SMACConfig, seed: int) -> None:
        super().__init__(config, seed)

        self.env = StarCraft2Env(map_name=self.task, seed=self.seed)

        self.env_info = self.env.get_env_info()

        self.possible_agents = [f"agent_{i}" for i in range(self.env_info["n_agents"])]

        self.observation: MARLObservation

        self.reset()

    @cached_property
    def max_action_value(self) -> dict[str, np.ndarray]:
        max_action_values = {}
        for agent_id in self.possible_agents:
            max_action_values[agent_id] = np.array(1.0)
        return max_action_values

    @cached_property
    def min_action_value(self) -> dict[str, np.ndarray]:
        min_action_values = {}
        for agent_id in self.possible_agents:
            min_action_values[agent_id] = np.array(0.0)
        return min_action_values

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        observation_space: dict[str, Any] = {}

        obs_dict = {}
        for agent_id in self.possible_agents:
            obs_dict[agent_id] = self.env_info["obs_shape"]

        observation_space["obs"] = obs_dict

        observation_space["state"] = self.env_info["state_shape"]
        observation_space["num_agents"] = self.env_info["n_agents"]

        return observation_space

    @cached_property
    def num_agents(self) -> int:
        return self.env_info["n_agents"]

    @cached_property
    def action_num(self) -> int:
        return self.env_info["n_actions"]

    def get_available_actions(self) -> dict[str, np.ndarray]:
        actions = {}
        for agent_index, agent_id in enumerate(self.possible_agents):
            available_actions = self.env.get_avail_agent_actions(agent_index)
            actions[agent_id] = available_actions
        return actions

    def sample_action(self) -> dict[str, np.ndarray]:
        actions = {}
        for agent_index, agent_id in enumerate(self.possible_agents):
            available_actions = self.env.get_avail_agent_actions(agent_index)
            available_actions_ind = np.nonzero(available_actions)[0]
            action = np.random.choice(available_actions_ind)
            actions[agent_id] = np.array(action)
        return actions  # type: ignore

    def set_seed(self, seed: int) -> None:
        self.env = StarCraft2Env(map_name=self.task, seed=seed)

        self.env_info = self.env.get_env_info()

        self.reset()

    def reset(self, training: bool = True) -> MARLObservation:
        obs, state = self.env.reset()

        # Convert obs list → dict[str -> obs_i]
        obs_dict = {agent_id: obs[i] for i, agent_id in enumerate(self.possible_agents)}

        self.observation = MARLObservation(
            global_state=state,
            agent_states=obs_dict,
            available_actions=self.get_available_actions(),
        )

        return self.observation

    def step(self, action: dict[str, np.ndarray]) -> MultiAgentExperience:  # type: ignore[override]

        ordered_actions = [int(action[agent_id]) for agent_id in self.possible_agents]

        reward, done, info = self.env.step(ordered_actions)

        obs = self.env.get_obs()
        # Convert obs list → dict[str -> obs_i]
        obs_dict = {agent_id: obs[i] for i, agent_id in enumerate(self.possible_agents)}

        next_observation = MARLObservation(
            global_state=self.env.get_state(),
            agent_states=obs_dict,
            available_actions=self.get_available_actions(),
        )

        rewards: dict[str, float] = {}
        dones: dict[str, bool] = {}
        for agent_id in self.possible_agents:
            rewards[agent_id] = reward
            dones[agent_id] = done

        experience = MultiAgentExperience(
            observation=self.observation,
            action=action,  # type: ignore
            reward=rewards,
            next_observation=next_observation,
            done=dones,
            truncated=dones,
            info=info,
        )

        self.observation = next_observation

        return experience

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render(mode="rgb_array")
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
