import abc
from functools import cached_property
from typing import Any

import numpy as np
from cares_reinforcement_learning.types.experience import MultiAgentExperience
from cares_reinforcement_learning.types.observation import MARLObservation
from cares_reinforcement_learning.envs.environments.base_environment import (
    BaseEnvironment,
)
from cares_reinforcement_learning.envs.util.configurations import GymEnvironmentConfig


class MARLEnvironment(BaseEnvironment[MARLObservation]):
    """
    Multi-Agent Reinforcement Learning Environment Base Class

    This class provides the interface for multi-agent environments where
    multiple agents interact simultaneously in a shared environment.
    """

    def __init__(self, config: GymEnvironmentConfig, seed: int) -> None:
        super().__init__(config, seed)

    @cached_property
    @abc.abstractmethod
    def min_action_value(self) -> list[np.ndarray]:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def max_action_value(self) -> list[np.ndarray]:
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def observation_space(self) -> dict[str, Any]:
        """
        Returns observation space information for multi-agent environment.
        Should include per-agent observation shapes and global state info.
        """
        raise NotImplementedError("Override this method")

    @cached_property
    @abc.abstractmethod
    def action_num(self) -> int:
        """Number of possible actions per agent"""
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def sample_action(self) -> list[int] | list[np.ndarray]:
        """
        Sample random actions for all agents.
        Returns: List of actions, one per agent
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def set_seed(self, seed: int) -> None:
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def reset(self, training: bool = True) -> MARLObservation:
        """
        Reset environment and return initial global state.
        Returns: Initial global state
        """
        raise NotImplementedError("Override this method")

    @abc.abstractmethod
    def step(self, action: list[int] | list[np.ndarray]) -> MultiAgentExperience:
        raise NotImplementedError("Override this method")

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        return np.zeros((height, width, 3), dtype=np.uint8)
