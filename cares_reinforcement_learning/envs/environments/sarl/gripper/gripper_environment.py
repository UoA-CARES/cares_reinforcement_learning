from functools import cached_property

import cv2
import numpy as np
from gripper_gym.environments.environment_factory import EnvironmentFactory

from cares_reinforcement_learning.envs.environments.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.util.configurations import GripperConfig
from cares_reinforcement_learning.util import helpers as hlp


class GripperEnvironment(SARLEnvironment):
    def __init__(
        self, config: GripperConfig, seed: int, image_observation: bool
    ) -> None:
        super().__init__(config, seed, image_observation)

        factory = EnvironmentFactory()
        self.domain = config.domain
        self.task = config.task
        self.gripper_id = config.gripper_id

        self.env = factory.create_environment(self.domain, self.task, self.gripper_id)
        self.set_seed(self.seed)

    @cached_property
    def min_action_value(self) -> np.ndarray:
        return self.env.min_action_value

    @cached_property
    def max_action_value(self) -> np.ndarray:
        return self.env.max_action_value

    @cached_property
    def _vector_space(self) -> int:
        observation_space = len(self.env.reset())
        return observation_space

    @cached_property
    def action_num(self) -> int:
        action_num = self.env.gripper.num_motors
        return action_num

    def sample_action(self):
        action = self.env.sample_action()
        return hlp.normalize(action, self.max_action_value, self.min_action_value)

    def set_seed(self, seed: int) -> None:
        if hasattr(self.env, "set_seed"):
            self.env.set_seed(seed)

    def _reset(self, training: bool = True):
        return self.env.reset()

    def _step(self, action):
        action = hlp.denormalize(action, self.max_action_value, self.min_action_value)
        return self.env.step(action)

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        if hasattr(self.env, "render"):
            frame = self.env.render()
        elif hasattr(self.env, "grab_frame"):
            frame = self.env.grab_frame()
        else:
            return np.zeros((height, width, 3), dtype=np.uint8)

        if frame is not None:
            frame = cv2.resize(frame, (width, height))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            return frame
        return np.zeros((height, width, 3), dtype=np.uint8)

    def get_overlay_info(self) -> dict:
        if hasattr(self.env, "get_overlay_info"):
            return self.env.get_overlay_info()
        return {}
