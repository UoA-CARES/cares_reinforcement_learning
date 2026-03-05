from functools import cached_property
from typing import Any

import cv2
import numpy as np
from smacv2.env.starcraft2.wrapper import StarCraftCapabilityEnvWrapper

from cares_reinforcement_learning.envs.marl.marl_environment import (
    MARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import SMAC2Config
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation


class SMAC2Environment(MARLEnvironment):
    def __init__(self, config: SMAC2Config, seed: int) -> None:
        super().__init__(config, seed)

        self.distribution_config = {
            "n_units": config.n_units,
            "n_enemies": config.n_enemies,
            "team_gen": {
                "dist_type": "weighted_teams",
                "unit_types": ["marine"],
                "weights": [1.0],
                "observe": True,
            },
            "start_positions": {
                "dist_type": "surrounded_and_reflect",
                "p": 0.5,
                "n_enemies": 3,
                "map_x": 32,
                "map_y": 32,
            },
        }

        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=self.distribution_config,
            map_name=self.task,
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            seed=self.seed,
        )

        self.env_info = self.env.get_env_info()

        self.agent_ids = [f"agent_{i}" for i in range(self.env_info["n_agents"])]

        self.observation: MARLObservation

        self.reset()

    @cached_property
    def max_action_value(self) -> list[np.ndarray]:
        max_action_values = []
        for _ in range(self.env_info["n_agents"]):
            max_action_values.append(np.array(1.0))
        return max_action_values

    @cached_property
    def min_action_value(self) -> list[np.ndarray]:
        min_action_values = []
        for _ in range(self.env_info["n_agents"]):
            min_action_values.append(np.array(0.0))
        return min_action_values

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        observation_space: dict[str, Any] = {}

        obs_dict = {}
        for agent_id in self.agent_ids:
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

    def get_available_actions(self) -> np.ndarray:
        actions = []
        for agent_id in range(self.env_info["n_agents"]):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            actions.append(avail_actions)
        return np.array(actions)

    def sample_action(self) -> list[int]:
        actions = []
        for agent_id in range(self.env_info["n_agents"]):
            avail_actions = self.env.get_avail_agent_actions(agent_id)
            avail_actions_ind = np.nonzero(avail_actions)[0]
            action = np.random.choice(avail_actions_ind)
            actions.append(action)
        return actions

    def set_seed(self, seed: int) -> None:
        self.env = StarCraftCapabilityEnvWrapper(
            capability_config=self.distribution_config,
            map_name="10gen_terran",
            debug=False,
            conic_fov=False,
            obs_own_pos=True,
            use_unit_ranges=True,
            min_attack_range=2,
            seed=seed,
        )

        self.env_info = self.env.get_env_info()

        self.reset()

    def reset(self, training: bool = True) -> MARLObservation:
        obs, state = self.env.reset()

        # Convert obs list → dict[str -> obs_i]
        obs_dict = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}

        self.observation = MARLObservation(
            global_state=state,
            agent_states=obs_dict,
            avail_actions=self.env.get_avail_actions(),
        )

        return self.observation

    def step(self, action: list[int]) -> MultiAgentExperience:  # type: ignore[override]
        reward, done, info = self.env.step(action)

        obs = self.env.get_obs()
        # Convert obs list → dict[str -> obs_i]
        obs_dict = {agent_id: obs[i] for i, agent_id in enumerate(self.agent_ids)}

        next_observation = MARLObservation(
            global_state=self.env.get_state(),
            agent_states=obs_dict,
            avail_actions=self.env.get_avail_actions(),
        )

        rewards = [0.0] * self.env_info["n_agents"]
        rewards[0] = reward  # Assuming reward is for all agents equally
        dones = [done] * self.env_info["n_agents"]

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
