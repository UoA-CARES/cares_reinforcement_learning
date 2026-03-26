from functools import cached_property

import numpy as np
from reinforcement_learning.EnvironmentFactory import EnvironmentFactory

from cares_reinforcement_learning.envs.configurations import F1TenthConfig
from cares_reinforcement_learning.envs.sarl.sarl_environment import SARLEnvironment
from cares_reinforcement_learning.util import helpers as hlp


class F1TenthEnvironment(SARLEnvironment):
    def __init__(self, config: F1TenthConfig, seed: int) -> None:
        super().__init__(config, seed, False)

        self.factory = EnvironmentFactory()

        # # Instantiate the task
        self.env = self.factory.create(config.task, {})

        self.set_seed(self.seed)

    def _reset(self, training: bool = True):
        state = self.env.reset()
        return state

    def sample_action(self):
        action = np.random.uniform(
            self.min_action_value, self.max_action_value, size=self.action_num
        )
        return hlp.normalize(action, self.max_action_value, self.min_action_value)

    def set_seed(self, seed: int) -> None:
        self.env.set_seed(seed)

    def get_overlay_info(self) -> dict:
        return {}

    def _step(self, action):
        action = hlp.denormalize(action, self.max_action_value, self.min_action_value)
        return self.env.step(action)

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.max_actions

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.min_actions

    @cached_property
    def _vector_space(self) -> int:
        return self.env.observation_size

    @cached_property
    def action_num(self) -> int:
        return self.env.action_num

    def grab_frame(self, height=720, width=1280) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)
