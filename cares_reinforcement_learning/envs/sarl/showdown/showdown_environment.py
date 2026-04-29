import os
import time
from functools import cached_property

import cv2
import numpy as np
from gymnasium import spaces
from showdown_gym.showdown_environment import SingleShowdownWrapper

from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import ShowdownConfig


class ShowdownEnvironment(SARLEnvironment):
    def __init__(
        self,
        config: ShowdownConfig,
        seed: int,
        image_observation: bool,
        evaluation: bool = False,
    ) -> None:
        super().__init__(config, seed, image_observation)

        # "random", "uber", "ou", "uu", "ru", "nu"
        team_type: str = config.domain

        # "max", "simple" or "random"
        opponent_type: str = config.task

        self.env = SingleShowdownWrapper(
            team_type=team_type,
            opponent_type=opponent_type,
            evaluation=evaluation,
        )
        time.sleep(3)  # Allow the environment to initialize properly

        self.set_seed(self.seed)

    def set_log_path(self, log_path: str, step_count: int) -> None:
        path = f"{log_path}/replays/{step_count}"
        if not os.path.exists(path):
            os.makedirs(path)
        self.env.env.agent1._save_replays = path

    @cached_property
    def max_action_value(self) -> float:
        return self.env.action_space.high[0]

    @cached_property
    def min_action_value(self) -> float:
        return self.env.action_space.low[0]

    @cached_property
    def _vector_space(self) -> int:
        return self.env.observation_space.shape[0]

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.env.action_space, spaces.Box):
            action_num = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_num = self.env.action_space.n
        else:
            raise ValueError(
                f"Unhandled action space type: {type(self.env.action_space)}"
            )
        return action_num

    def sample_action(self) -> int:
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        _, _ = self.env.reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.env.action_space.seed(seed)

    def _reset(self, training: bool = True) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def _step(self, action: int) -> tuple:
        state, reward, done, truncated, info = self.env.step(np.int64(action))
        return state, reward, done, truncated, info

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame = self.env.render()
        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
