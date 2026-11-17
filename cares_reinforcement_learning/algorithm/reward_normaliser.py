from typing import Any

import numpy as np
import torch


class RunningRewardNormaliser:
    """
    Track discounted-return statistics for SimbaV2 reward scaling.

    Raw rewards remain stored in replay. Running return statistics are
    updated once per environment interaction, while sampled replay rewards
    are scaled using the current statistics.
    """

    def __init__(
        self,
        gamma: float,
        g_max: float = 5.0,
        count_epsilon: float = 1e-4,
        normalisation_epsilon: float = 1e-8,
    ):
        self.gamma = gamma
        self.g_max = g_max
        self.normalisation_epsilon = normalisation_epsilon

        self.discounted_return = 0.0

        self.mean = 0.0
        self.var = 1.0
        self.count = float(count_epsilon)

        self.max_abs_return = 0.0

    def update(
        self,
        reward: float,
        done: bool,
    ) -> None:
        reward = float(reward)

        self.discounted_return = self.gamma * self.discounted_return + reward

        self._update_from_value(self.discounted_return)

        self.max_abs_return = max(
            self.max_abs_return,
            abs(self.discounted_return),
        )

        if done:
            self.discounted_return = 0.0

    def _update_from_value(self, value: float) -> None:
        batch_count = 1.0

        delta = value - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        current_m2 = self.var * self.count
        correction = delta**2 * self.count * batch_count / total_count

        new_var = (current_m2 + correction) / total_count

        self.mean = float(new_mean)
        self.var = float(new_var)
        self.count = float(total_count)

    def _get_scale(self) -> float:
        variance_scale = np.sqrt(self.var + self.normalisation_epsilon)

        return_bound_scale = self.max_abs_return / self.g_max

        return float(
            max(
                variance_scale,
                return_bound_scale,
                self.normalisation_epsilon,
            )
        )

    def normalise_tensor(
        self,
        reward: torch.Tensor,
    ) -> torch.Tensor:
        return reward / self._get_scale()

    def state_dict(self) -> dict[str, Any]:
        return {
            "discounted_return": float(self.discounted_return),
            "mean": float(self.mean),
            "var": float(self.var),
            "count": float(self.count),
            "max_abs_return": float(self.max_abs_return),
        }

    def load_state_dict(
        self,
        state_dict: dict[str, Any],
    ) -> None:
        self.discounted_return = float(state_dict["discounted_return"])
        self.mean = float(state_dict["mean"])
        self.var = float(state_dict["var"])
        self.count = float(state_dict["count"])
        self.max_abs_return = float(state_dict["max_abs_return"])

    def get_statistics(self) -> dict[str, float]:
        return {
            "reward_norm_count": self.count,
            "reward_norm_return_mean": self.mean,
            "reward_norm_return_var": self.var,
            "reward_norm_max_abs_return": self.max_abs_return,
            "reward_norm_scale": self._get_scale(),
        }
