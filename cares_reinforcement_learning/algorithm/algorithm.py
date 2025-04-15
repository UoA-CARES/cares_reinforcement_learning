"""
Original Paper: https://arxiv.org/abs/1802.09477v3

"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch

from cares_reinforcement_learning.memory import MemoryBuffer


class Algorithm(ABC):
    def __init__(
        self,
        policy_type: Literal["value", "policy", "discrete_policy", "mbrl"],
        device: torch.device,
    ):
        self.policy_type = policy_type
        self.device = device

    @abstractmethod
    def select_action_from_policy(
        self, state: Any, evaluation: bool = False
    ) -> np.ndarray | float | tuple[np.ndarray, np.ndarray]: ...

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
        **kwargs: Any,
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
    ) -> np.ndarray | float | tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Implement in base class")


class ImageAlgorithm(Algorithm):

    @abstractmethod
    def select_action_from_policy(
        self, state: dict[str, np.ndarray], evaluation: bool = False
    ) -> np.ndarray | float | tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Implement in base class")
