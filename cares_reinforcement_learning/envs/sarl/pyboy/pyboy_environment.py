from functools import cached_property

import numpy as np
from pyboy_environment import suite

from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import PyBoyConfig


class PyboyEnvironment(SARLEnvironment):
    def __init__(self, config: PyBoyConfig, seed: int, image_observation: bool) -> None:
        super().__init__(config, seed, image_observation)

        self.env = suite.make(
            config.domain,
            config.task,
            config.act_freq,
            config.emulation_speed,
            config.headless,
        )

        self.set_seed(self.seed)

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.min_action_value

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.max_action_value

    @cached_property
    def _vector_space(self) -> int:
        return self.env.observation_space

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def sample_action(self) -> int:
        return self.env.sample_action()

    def set_seed(self, seed: int) -> None:
        self.env.set_seed(seed)

    def _reset(self, training: bool = True) -> np.ndarray:
        return self.env.reset(training=training)

    def _step(self, action: int) -> tuple:
        return self.env.step(action)

    def grab_frame(self, height=240, width=300) -> np.ndarray:
        return self.env.grab_frame(height, width)

    def get_overlay_info(self) -> dict:
        return self.env.get_overlay_info()
