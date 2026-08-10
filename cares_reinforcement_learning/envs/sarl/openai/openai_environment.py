from functools import cached_property
from typing import Any

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from cares_reinforcement_learning.envs.configurations import OpenAIConfig
from cares_reinforcement_learning.envs.sarl.sarl_environment import (
    SARLEnvironment,
)
from cares_reinforcement_learning.util import helpers as hlp


class OpenAIEnvironment(SARLEnvironment):
    def __init__(
        self, config: OpenAIConfig, seed: int, image_observation: bool
    ) -> None:
        super().__init__(config, seed, image_observation)

        self.env = gym.make(config.task, render_mode="rgb_array")
        self.set_seed(self.seed)

        # If Box space, we will apply action normalization - even if redundant
        self.apply_action_normalization = isinstance(self.env.action_space, spaces.Box)

        self.slippery_friction = config.slippery_friction
        self.slippery_switch_every = config.slippery_switch_every
        self.slippery_min_friction = config.slippery_min_friction
        self.slippery_max_friction = config.slippery_max_friction

        self.slippery_global_step = 0
        self.slippery_steps_since_change = 0
        self.slippery_change_pending = False
        self.slippery_current_friction: float | None = None
        self.slippery_base_friction: NDArray[np.float64] | None = None
        self.slippery_target_geom_indices: NDArray[np.int64] | None = None
        self.slippery_rng = np.random.default_rng(seed)

        if self.slippery_friction:
            self._setup_slippery_friction()

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

    def sample_action(self) -> int | np.ndarray:
        action = self.env.action_space.sample()
        if self.apply_action_normalization:
            action = hlp.normalize(action, self.max_action_value, self.min_action_value)
        return action

    def set_seed(self, seed: int) -> None:
        _, _ = self.env.reset(seed=seed)
        # Note issues: https://github.com/rail-berkeley/softlearning/issues/75
        self.env.action_space.seed(seed)

    def _reset(self, training: bool = True) -> np.ndarray:
        self._maybe_apply_slippery_change_on_reset()
        state, _ = self.env.reset()
        return state

    def _step(self, action: np.ndarray) -> tuple:
        if self.apply_action_normalization:
            action = hlp.denormalize(
                action, self.max_action_value, self.min_action_value
            )

        state, reward, done, truncated, info = self.env.step(action)

        if self.slippery_friction:
            self.slippery_global_step += 1
            self.slippery_steps_since_change += 1
            self._maybe_mark_slippery_change_pending()

            info = dict(info)
            info["slippery_friction"] = self.slippery_current_friction
            info["slippery_change_pending"] = self.slippery_change_pending
            info["slippery_steps_since_change"] = self.slippery_steps_since_change

        return state, reward, done, truncated, info

    def grab_frame(self, height: int = 240, width: int = 300) -> np.ndarray:
        frame: np.ndarray = self.env.render()  # type: ignore

        frame = cv2.resize(frame, (width, height))
        # Convert to BGR for use with OpenCV
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def get_overlay_info(self) -> dict[str, Any]:
        # TODO: Add overlay information for gyms as needed
        if not self.slippery_friction:
            return {}

        return {
            "slippery_friction": self.slippery_current_friction,
            "slippery_steps_since_change": self.slippery_steps_since_change,
            "slippery_change_pending": self.slippery_change_pending,
        }

    def _setup_slippery_friction(self) -> None:
        if not hasattr(self.env.unwrapped, "model"):
            raise TypeError("Slippery friction requires a MuJoCo environment.")

        model = self.env.unwrapped.model

        if not hasattr(model, "geom_friction"):
            raise TypeError("MuJoCo model does not expose geom_friction.")

        base_friction = np.asarray(model.geom_friction.copy(), dtype=np.float64)

        if base_friction.ndim != 2 or base_friction.shape[1] != 3:
            raise ValueError("Expected geom_friction to have shape [num_geoms, 3].")

        # MuJoCo geom_type == 0 is plane/floor.
        self.slippery_base_friction = base_friction
        self.slippery_target_geom_indices = np.asarray(
            [
                geom_id
                for geom_id in range(int(model.ngeom))
                if int(model.geom_type[geom_id]) != 0
            ],
            dtype=np.int64,
        )

        if self.slippery_target_geom_indices.size == 0:
            raise ValueError("No non-floor MuJoCo geoms found for slippery friction.")

        self.slippery_current_friction = self._sample_slippery_friction()
        self._apply_slippery_friction()

    def _sample_slippery_friction(self) -> float:
        log_min = np.log10(self.slippery_min_friction)
        log_max = np.log10(self.slippery_max_friction)
        return float(10.0 ** self.slippery_rng.uniform(log_min, log_max))

    def _apply_slippery_friction(self) -> None:
        if (
            self.slippery_base_friction is None
            or self.slippery_target_geom_indices is None
            or self.slippery_current_friction is None
        ):
            return

        model = self.env.unwrapped.model

        model.geom_friction[:, :] = self.slippery_base_friction
        model.geom_friction[self.slippery_target_geom_indices, 0] = (
            self.slippery_current_friction
        )

    def _maybe_mark_slippery_change_pending(self) -> None:
        if self.slippery_steps_since_change >= self.slippery_switch_every:
            self.slippery_change_pending = True

    def _maybe_apply_slippery_change_on_reset(self) -> bool:
        if not self.slippery_friction:
            return False

        changed = False

        if self.slippery_change_pending:
            self.slippery_current_friction = self._sample_slippery_friction()
            self.slippery_steps_since_change = 0
            self.slippery_change_pending = False
            changed = True

        self._apply_slippery_friction()
        return changed
