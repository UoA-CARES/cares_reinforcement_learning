"""
Original Paper: https://arxiv.org/abs/1802.09477v3

"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


class Algorithm(ABC):
    def __init__(
        self,
        policy_type: Literal["value", "policy", "discrete_policy", "mbrl"],
        config: AlgorithmConfig,
        device: torch.device,
    ):
        self.policy_type = policy_type

        self.gamma = config.gamma

        self.G = config.G
        self.number_steps_per_train_policy = config.number_steps_per_train_policy

        self.buffer_size = config.buffer_size
        self.batch_size = config.batch_size

        self.max_steps_exploration = config.max_steps_exploration
        self.max_steps_training = config.max_steps_training

        self.image_observation = config.image_observation

        self.device = device

    @abstractmethod
    def select_action_from_policy(
        self, state: Any, evaluation: bool = False
    ) -> int | np.ndarray: ...

    @abstractmethod
    def calculate_bias(
        self,
        episode_states: list[Any],
        episode_actions: list[np.ndarray | int],
        episode_rewards: list[float],
    ) -> dict[str, Any]: ...

    # TODO push batch_size into the algorithm
    @abstractmethod
    def train_policy(
        self, memory: MemoryBuffer, batch_size: int, training_step: int
    ) -> dict[str, Any]: ...

    @abstractmethod
    def save_models(self, filepath: str, filename: str) -> None: ...

    @abstractmethod
    def load_models(self, filepath: str, filename: str) -> None: ...

    def get_intrinsic_reward(
        self,
        state: dict[str, np.ndarray],  # pylint: disable=unused-argument
        action: np.ndarray,  # pylint: disable=unused-argument
        next_state: dict[str, np.ndarray],  # pylint: disable=unused-argument
        **kwargs: Any,  # pylint: disable=unused-argument
    ) -> float:
        """
        Calculate intrinsic reward based on the state, action, and next state.
        This is a placeholder method and should be implemented in subclasses if needed.
        """
        return 0.0


class VectorAlgorithm(Algorithm):

    @abstractmethod
    def select_action_from_policy(
        self, state: np.ndarray, evaluation: bool = False
    ) -> int | np.ndarray: ...

    @abstractmethod
    def calculate_bias(
        self,
        episode_states: list[np.ndarray],
        episode_actions: list[np.ndarray | int],
        episode_rewards: list[float],
    ) -> dict[str, Any]: ...


class ImageAlgorithm(Algorithm):

    @abstractmethod
    def select_action_from_policy(
        self, state: dict[str, np.ndarray], evaluation: bool = False
    ) -> int | np.ndarray: ...

    @abstractmethod
    def calculate_bias(
        self,
        episode_states: list[dict[str, np.ndarray]],
        episode_actions: list[np.ndarray | int],
        episode_rewards: list[float],
    ) -> dict[str, Any]: ...
