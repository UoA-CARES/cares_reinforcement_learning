import abc
from collections import deque
from functools import cached_property
from typing import Any

import cv2
import numpy as np

from cares_reinforcement_learning.envs.environments.base_environment import (
    BaseEnvironment,
)
from cares_reinforcement_learning.envs.util.configurations import GymEnvironmentConfig
from cares_reinforcement_learning.types.experience import SingleAgentExperience
from cares_reinforcement_learning.types.observation import SARLObservation


class SARLEnvironment(BaseEnvironment[SARLObservation]):
    def __init__(
        self, config: GymEnvironmentConfig, seed: int, image_observation: bool
    ) -> None:
        super().__init__(config, seed)

        self.state_std = config.state_std
        self.action_std = config.action_std

        self.image_observation = image_observation

        self.grey_scale = bool(config.grey_scale)

        self.frames_to_stack = config.frames_to_stack
        self.frames_stacked: deque[np.ndarray] = deque([], maxlen=self.frames_to_stack)

        self.frame_width = config.frame_width
        self.frame_height = config.frame_height

        self.observation: SARLObservation

    @abc.abstractmethod
    def get_overlay_info(self) -> dict:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def min_action_value(self) -> np.ndarray:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def max_action_value(self) -> np.ndarray:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def _vector_space(self) -> int:
        raise NotImplementedError("Override this method")

    @cached_property
    def observation_space(self) -> dict[str, Any]:
        channels = 1 if self.grey_scale else 3
        channels *= self.frames_to_stack
        image_space = (channels, self.frame_height, self.frame_width)

        vector_space = self._vector_space

        return {"image": image_space, "vector": vector_space}

    @cached_property
    @abc.abstractmethod
    def action_num(self) -> int:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def sample_action(self):
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    def _image_state(self, reset: bool = False) -> np.ndarray:
        frame = self.grab_frame(height=self.frame_height, width=self.frame_width)

        if self.grey_scale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = frame.reshape((self.frame_height, self.frame_width, 1))

        frame = np.moveaxis(frame, -1, 0)
        if reset:
            for _ in range(self.frames_to_stack - 1):
                self.frames_stacked.append(frame)

        self.frames_stacked.append(frame)
        image_state = np.concatenate(list(self.frames_stacked), axis=0)

        return image_state

    def _add_relative_noise(
        self, data: np.ndarray, rel_std: float, min_std: float = 1e-3
    ) -> np.ndarray:
        """
        Adds Gaussian noise proportional to the absolute value of each element.
        rel_std = fraction of magnitude to perturb (e.g., 0.02 = 2%)
        min_std = lower bound to prevent zero noise for small values
        """
        # Per-element scale (avoid zeros)
        sigma = np.maximum(np.abs(data) * rel_std, min_std)

        # Gaussian noise with proportional std
        noise = np.random.normal(0, sigma, size=data.shape)
        return data + noise

    @abc.abstractmethod
    def _reset(self, training: bool = True) -> np.ndarray:
        raise NotImplementedError("Override this method")

    def reset(self, training: bool = True) -> SARLObservation:
        state = self._reset(training=training)

        image_state = self._image_state(reset=True) if self.image_observation else None

        self.observation = SARLObservation(vector_state=state, image_state=image_state)
        return self.observation

    @abc.abstractmethod
    def _step(self, action):
        raise NotImplementedError("Override this method")

    def step(self, action: np.ndarray) -> SingleAgentExperience:
        # Apply action noise
        action_noise = action
        if self.action_std > 0:
            action_noise = self._add_relative_noise(action, self.action_std)
            action_noise = np.clip(
                action_noise, self.min_action_value, self.max_action_value
            )

        # Execute environment step (existing logic)
        vector_state, reward, done, truncated, info = self._step(action_noise)
        image_state = self._image_state() if self.image_observation else None

        # Apply observation noise
        if self.state_std > 0:
            vector_state = self._add_relative_noise(vector_state, self.state_std)

        next_observation = SARLObservation(
            vector_state=vector_state, image_state=image_state
        )

        experience = SingleAgentExperience(
            observation=self.observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
            truncated=truncated,
            info=info,
        )

        self.observation = next_observation

        return experience

    @abc.abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")
