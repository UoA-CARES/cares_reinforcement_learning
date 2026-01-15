from dataclasses import dataclass
from typing import Any

import numpy as np

from cares_reinforcement_learning.types.observation import Observation


@dataclass
class Experience:
    state: Observation
    next_state: Observation


@dataclass
class SingleAgentExperience(Experience):
    action: np.ndarray | int
    reward: float
    done: bool
    truncated: bool

    @property
    def done_flag(self) -> bool:
        return self.done

    @property
    def truncated_flag(self) -> bool:
        return self.truncated

    @property
    def reward_sum(self) -> float:
        return self.reward


@dataclass
class MultiAgentExperience(Experience):
    action: dict[str, np.ndarray] | dict[str, int]
    reward: dict[str, float]
    done: dict[str, bool]
    truncated: dict[str, bool]

    @property
    def done_flag(self) -> bool:
        # terminal when *all* agents are done
        return all(self.done.values())

    @property
    def truncated_flag(self) -> bool:
        # truncated when *all* agents are truncated
        return all(self.truncated.values())

    @property
    def reward_sum(self) -> float:
        return float(sum(self.reward.values()))
