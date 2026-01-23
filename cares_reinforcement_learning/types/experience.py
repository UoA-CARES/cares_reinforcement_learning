from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    Observation,
    SARLObservation,
)

ObsType = TypeVar("ObsType", bound=Observation)


@dataclass(frozen=True, slots=True)
class Experience(Generic[ObsType]):
    observation: ObsType
    next_observation: ObsType

    info: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SingleAgentExperience(Experience[SARLObservation]):
    action: np.ndarray
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

    def clone(self) -> "SingleAgentExperience":
        return SingleAgentExperience(
            observation=self.observation.clone(),
            action=self.action.copy(),
            reward=float(self.reward),
            next_observation=self.next_observation.clone(),
            done=bool(self.done),
            truncated=bool(self.truncated),
            info=self.info.copy(),
        )


@dataclass(frozen=True, slots=True)
class MultiAgentExperience(Experience[MARLObservation]):
    action: list[np.ndarray]
    reward: list[float]
    done: list[bool]
    truncated: list[bool]

    @property
    def done_flag(self) -> bool:
        # terminal when *all* agents are done
        return all(self.done)

    @property
    def truncated_flag(self) -> bool:
        # truncated when *all* agents are truncated
        return all(self.truncated)

    @property
    def reward_sum(self) -> float:
        return float(sum(self.reward))


ExperienceType = SingleAgentExperience | MultiAgentExperience
