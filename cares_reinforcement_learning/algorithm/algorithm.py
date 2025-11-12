"""
Original Paper: https://arxiv.org/abs/1802.09477v3

"""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
import torch

import cares_reinforcement_learning.util.helpers as hlp
from cares_reinforcement_learning.util.configurations import AlgorithmConfig
from cares_reinforcement_learning.util.training_context import (
    TrainingContext,
    ActionContext,
)


class Algorithm(ABC):
    def __init__(
        self,
        policy_type: Literal["value", "policy", "discrete_policy", "mbrl", "usd"],
        config: AlgorithmConfig,
        device: torch.device,
    ):
        self.policy_type: Literal[
            "value", "policy", "discrete_policy", "mbrl", "usd"
        ] = policy_type

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
        self, action_context: ActionContext
    ) -> int | np.ndarray: ...

    def _fixed_step_bias_segments(
        self, values: list[float], step_boundaries: list[int] | None = None
    ) -> dict[str, dict[str, float]]:

        if step_boundaries is None:
            step_boundaries = [0, 100, 200, 300, 400]

        episode_length = len(values)
        segment_stats = {}

        for i, start in enumerate(step_boundaries):
            end = (
                step_boundaries[i + 1]
                if i + 1 < len(step_boundaries)
                else episode_length
            )
            if start >= episode_length:
                continue  # skip if start beyond episode length

            segment = values[start : min(end, episode_length)]
            if len(segment) == 0:
                continue

            key = f"bias_segment_{start}_{end if end < episode_length else 'end'}"
            segment_stats[key] = {
                "mean": float(np.mean(segment)),
                "abs_mean": float(np.mean(np.abs(segment))),
                "std": float(np.std(segment)),
            }

        return segment_stats

    # @abstractmethod
    def _calculate_value(
        self, state: Any, action: int | np.ndarray
    ) -> float:  # pylint: disable=unused-argument
        return 0.0

    def calculate_bias(
        self,
        episode_states: Any,
        episode_actions: list[int | np.ndarray],
        episode_rewards: list[float],
    ) -> dict[str, Any]:

        discounted_returns = hlp.compute_discounted_returns(episode_rewards, self.gamma)
        average_discounted_return = np.mean(discounted_returns)

        biases = []
        biases_normalised = []
        for state, action, discounted_return in zip(
            episode_states, episode_actions, discounted_returns
        ):
            value = self._calculate_value(state, action)

            bias = value - discounted_return
            biases.append(bias)

            bias_normalised = bias / (max(np.abs(average_discounted_return), 1e-8))
            biases_normalised.append(bias_normalised)

        bias_mean = np.mean(biases)
        bias_abs_mean = np.mean(np.abs(biases))
        bias_std = np.std(biases)

        bias_mean_normalised = np.mean(biases_normalised)
        bias_abs_mean_normalised = np.mean(np.abs(biases_normalised))
        bias_std_normalised = np.std(biases_normalised)

        bias_segments = self._fixed_step_bias_segments(biases_normalised)

        info = {
            "bias_mean": bias_mean,
            "bias_abs_mean": bias_abs_mean,
            "bias_std": bias_std,
            "bias_mean_normalised": bias_mean_normalised,
            "bias_abs_mean_normalised": bias_abs_mean_normalised,
            "bias_std_normalised": bias_std_normalised,
            "average_discounted_return": average_discounted_return,
            "bias_segments": bias_segments,
        }
        return info

    # TODO push batch_size into the algorithm
    @abstractmethod
    def train_policy(self, training_context: TrainingContext) -> dict[str, Any]: ...

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

    def episode_done(self):
        """
        This method is called when an episode is done.
        It can be overridden in subclasses to perform any necessary cleanup or logging.
        """
        pass


class VectorAlgorithm(Algorithm):

    @abstractmethod
    def select_action_from_policy(
        self, action_context: ActionContext
    ) -> int | np.ndarray: ...


class ImageAlgorithm(Algorithm):

    @abstractmethod
    def select_action_from_policy(
        self, action_context: ActionContext
    ) -> int | np.ndarray: ...
