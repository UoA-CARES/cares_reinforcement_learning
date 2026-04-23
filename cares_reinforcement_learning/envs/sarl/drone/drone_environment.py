from functools import cached_property
from typing import cast

import numpy as np
from drone_gym import task_factory
from typing_extensions import Literal

from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import DroneConfig
from cares_reinforcement_learning.util import helpers as hlp


class DroneEnvironment(SARLEnvironment):
    def __init__(self, config: DroneConfig, seed: int, image_observation: bool) -> None:
        super().__init__(config, seed, image_observation)

        if config.use_simulator not in [0, 1]:
            raise ValueError("use_simulator must be 0 (real drone) or 1 (simulator)")

        # Instantiate the task
        self.env = task_factory.make(
            config.task, use_simulator=cast(Literal[0, 1], config.use_simulator)
        )

        self.set_seed(self.seed)

    def _reset(self, training: bool = True) -> np.ndarray:
        return self.env.reset(training)

    def sample_action(self) -> np.ndarray:
        action = self.env.sample_action()
        return hlp.normalize(action, self.max_action_value, self.min_action_value)

    def set_seed(self, seed: int) -> None:
        self.env.set_seed(seed)

    def get_overlay_info(self) -> dict:
        return self.env.get_overlay_info()

    def _step(self, action) -> tuple:
        action = hlp.denormalize(action, self.max_action_value, self.min_action_value)
        return self.env.step(action)

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.max_action_value

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.min_action_value

    @cached_property
    def _vector_space(self) -> int:
        return self.env.observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def grab_frame(self, height=720, width=1280) -> np.ndarray:
        return self.env.grab_frame(height, width)
