from functools import cached_property

import numpy as np
from url_gym import task_factory

from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import URLConfig
from cares_reinforcement_learning.util import helpers as hlp


class URLEnvironment(SARLEnvironment):
    def __init__(self, config: URLConfig, seed: int, image_observation: bool) -> None:
        super().__init__(config, seed, image_observation)

        extra_kwargs = None
        if config.max_episode_steps is not None:
            extra_kwargs = {"max_episode_steps": config.max_episode_steps}

        # Instantiate the task in its native (un-normalized) action space
        self.env = task_factory.make(
            config.domain, config.task, extra_kwargs=extra_kwargs
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

    def grab_frame(self, height=240, width=300) -> np.ndarray:
        return self.env.grab_frame(height, width)
