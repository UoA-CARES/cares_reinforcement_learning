import logging
from functools import cached_property

import cv2
import numpy as np
from dm_control import suite

from cares_reinforcement_learning.envs.environments.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.util.configurations import DMCSConfig
from cares_reinforcement_learning.util import helpers as hlp


class DMCSEnvironment(SARLEnvironment):
    def __init__(self, config: DMCSConfig, seed: int, image_observation: bool) -> None:
        super().__init__(config, seed, image_observation)
        logging.info(f"Training on Domain {config.domain}")

        self.domain = config.domain
        self.env = suite.load(self.domain, self.task, task_kwargs={"random": self.seed})

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.action_spec().minimum

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.action_spec().maximum

    @cached_property
    def _vector_space(self) -> int:
        time_step = self.env.reset()
        # e.g. position, orientation, joint_angles
        observation = np.hstack(list(time_step.observation.values()))
        return len(observation)

    @cached_property
    def action_num(self) -> int:
        return self.env.action_spec().shape[0]

    def sample_action(self) -> np.ndarray:
        return np.random.uniform(
            self.min_action_value, self.max_action_value, size=self.action_num
        )

    def set_seed(self, seed: int) -> None:
        self.env = suite.load(self.domain, self.task, task_kwargs={"random": seed})

    def _reset(self, training: bool = True) -> np.ndarray:
        time_step = self.env.reset()
        observation = np.hstack(
            list(time_step.observation.values())
        )  # # e.g. position, orientation, joint_angles
        return observation

    def _step(self, action: np.ndarray) -> tuple:
        action = hlp.normalize(action, self.max_action_value, self.min_action_value)

        time_step = self.env.step(action)
        state, reward, done = (
            np.hstack(list(time_step.observation.values())),
            time_step.reward,
            time_step.last(),
        )
        # for consistency with open ai gym just add false for truncated
        return state, reward, done, False, {}

    def grab_frame(self, height=240, width=300, camera_id=0) -> np.ndarray:
        frame = self.env.physics.render(camera_id=camera_id, height=height, width=width)
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict:
        # TODO: Add overlay information for gyms as needed
        return {}
