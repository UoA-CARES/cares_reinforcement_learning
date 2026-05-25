from functools import cached_property

import cv2
import numpy as np
from gymnasium import spaces
import safety_gymnasium

from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.envs.configurations import SafetyGymnasiumConfig
from cares_reinforcement_learning.util import helpers as hlp


class SafetyGymnasiumEnvironment(SARLEnvironment):
    def __init__(
        self,
        config: SafetyGymnasiumConfig,
        seed: int,
        image_observation: bool
    ) -> None:
        super().__init__(config, seed, image_observation)

        self.env = safety_gymnasium.make(
            id=config.task,
            autoreset=False,
            render_mode="rgb_array"
        )
        self.set_seed(self.seed)

        # If Box space, we will apply action normalization - even if redundant
        self.apply_action_normalization = isinstance(self.env.action_space, spaces.Box)

    @cached_property
    def max_action_value(self) -> np.ndarray:
        if isinstance(self.env.action_space, spaces.Box):
            return self.env.action_space.high
        raise ValueError("Action space is not continuous")

    @cached_property
    def min_action_value(self) -> np.ndarray:
        if isinstance(self.env.action_space, spaces.Box):
            return self.env.action_space.low
        raise ValueError("Action space is not continuous")

    @cached_property
    def _vector_space(self) -> int:
        if self.env.observation_space.shape is not None:
            return self.env.observation_space.shape[0]
        raise ValueError("Observation space has not been set by gym")

    @cached_property
    def action_num(self) -> int:
        if isinstance(self.env.action_space, spaces.Box):
            action_num = self.env.action_space.shape[0]
        elif isinstance(self.env.action_space, spaces.Discrete):
            action_num = int(self.env.action_space.n)
        else:
            raise ValueError(
                f"Unhandled action space type: {type(self.env.action_space)}"
            )
        return action_num

    def set_seed(self, seed: int) -> None:
        self.env.reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.env.action_space.seed(seed)

    def _step(self, action: np.ndarray) -> tuple:
        if self.apply_action_normalization:
            action = hlp.denormalize(
                action, self.max_action_value, self.min_action_value
            )

        obs, reward, cost, terminated, truncated, info = self.env.step(action)
        info["cost"] = cost  # avoid signature change
        return obs, reward, terminated, truncated, info

    def _reset(self, training: bool = True) -> np.ndarray:
        state, _ = self.env.reset()
        return state

    def sample_action(self) -> int | np.ndarray:
        action = self.env.action_space.sample()
        if self.apply_action_normalization:
            action = hlp.normalize(action, self.max_action_value, self.min_action_value)
        return action

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame: np.ndarray = self.env.render()  # type: ignore

        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame