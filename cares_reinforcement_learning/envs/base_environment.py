import logging
from abc import ABC, abstractmethod
from functools import cached_property
from typing import Any, Generic, TypeVar

import cv2
import numpy as np

from cares_reinforcement_learning.envs.configurations import GymEnvironmentConfig
from cares_reinforcement_learning.types.experience import Experience
from cares_reinforcement_learning.types.observation import Observation

ObsType = TypeVar("ObsType", bound=Observation)


class BaseEnvironment(ABC, Generic[ObsType]):
    """
    Base Environment class for both single-agent and multi-agent environments.

    This class provides the common interface and functionality that both
    GymEnvironment and MARLEnvironment share, making it easier to handle
    both types uniformly in training loops and run scripts.
    """

    def __init__(self, config: GymEnvironmentConfig, seed: int) -> None:
        logging.info(f"Training with Task {config.task}")

        self.task = config.task

        self.seed = seed

    def render(self):
        frame = self.grab_frame()
        cv2.imshow(f"{self.task}", frame)
        cv2.waitKey(10)

    def set_log_path(self, log_path: str, step_count: int) -> None:
        pass

    @cached_property
    def num_agents(self) -> int:
        return 1

    @cached_property
    @abstractmethod
    def min_action_value(self) -> Any:
        raise NotImplementedError("Override this method")

    @cached_property
    @abstractmethod
    def max_action_value(self) -> Any:
        raise NotImplementedError("Override this method")

    @cached_property
    @abstractmethod
    def observation_space(self) -> dict[str, Any]:
        raise NotImplementedError("Override this method")

    @cached_property
    @abstractmethod
    def action_num(self) -> int:
        raise NotImplementedError("Override this method")

    @abstractmethod
    def reset(self, training: bool = True) -> ObsType:
        raise NotImplementedError("Override this method")

    @abstractmethod
    def step(self, action: Any) -> Experience[ObsType]:
        raise NotImplementedError("Override this method")

    @abstractmethod
    def sample_action(self) -> Any:
        raise NotImplementedError("Override this method")

    @abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    @abstractmethod
    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        raise NotImplementedError("Override this method")

    @abstractmethod
    def get_overlay_info(self) -> dict:
        raise NotImplementedError("Override this method")
