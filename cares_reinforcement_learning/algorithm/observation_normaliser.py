from typing import Any

import numpy as np
import torch


class RunningObservationNormaliser:
    """Track running observation mean/variance for SimBa RSNorm."""

    def __init__(
        self,
        shape: tuple[int, ...],
        count_epsilon: float = 1e-4,
        normalisation_epsilon: float = 1e-8,
    ):
        self.shape = shape

        # Float64 is useful here because these statistics accumulate
        # over potentially millions of environment interactions.
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(count_epsilon)

        self.normalisation_epsilon = normalisation_epsilon

    def update(self, observation: np.ndarray) -> None:
        """Update running statistics from one observation or a batch."""
        observation = np.asarray(observation, dtype=np.float64)

        # Single observation: [obs_dim] -> [1, obs_dim]
        if observation.shape == self.shape:
            observation = observation[np.newaxis, ...]

        if observation.shape[1:] != self.shape:
            raise ValueError(
                f"Expected observation shape {self.shape}, "
                f"got {observation.shape[1:]}"
            )

        batch_mean = np.mean(observation, axis=0)
        batch_var = np.var(observation, axis=0)
        batch_count = observation.shape[0]

        self._update_from_moments(
            batch_mean=batch_mean,
            batch_var=batch_var,
            batch_count=batch_count,
        )

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Parallel running mean/variance update."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count

        current_m2 = self.var * self.count
        batch_m2 = batch_var * batch_count

        correction = np.square(delta) * self.count * batch_count / total_count

        new_var = (current_m2 + batch_m2 + correction) / total_count

        self.mean = new_mean
        self.var = new_var
        self.count = float(total_count)

    def normalise(
        self,
        observation: np.ndarray,
    ) -> np.ndarray:
        """Normalise numpy observations using current statistics."""
        observation = np.asarray(observation)

        normalised = (observation - self.mean) / np.sqrt(
            self.var + self.normalisation_epsilon
        )

        return normalised.astype(observation.dtype, copy=False)

    def normalise_tensor(
        self,
        observation: torch.Tensor,
    ) -> torch.Tensor:
        """Normalise torch observations using current statistics."""
        mean = torch.as_tensor(
            self.mean, dtype=observation.dtype, device=observation.device
        )

        var = torch.as_tensor(
            self.var, dtype=observation.dtype, device=observation.device
        )

        return (observation - mean) / torch.sqrt(var + self.normalisation_epsilon)

    def state_dict(self) -> dict[str, Any]:
        return {
            "mean": torch.as_tensor(self.mean).cpu(),
            "var": torch.as_tensor(self.var).cpu(),
            "count": float(self.count),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.mean = state_dict["mean"].cpu().numpy().astype(np.float64)
        self.var = state_dict["var"].cpu().numpy().astype(np.float64)
        self.count = float(state_dict["count"])

    def get_statistics(self) -> dict[str, float]:
        return {
            "rsnorm_count": self.count,
            "rsnorm_mean_abs": float(np.mean(np.abs(self.mean))),
            "rsnorm_var_mean": float(np.mean(self.var)),
            "rsnorm_var_min": float(np.min(self.var)),
            "rsnorm_var_max": float(np.max(self.var)),
        }
