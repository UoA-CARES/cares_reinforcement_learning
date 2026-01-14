from dataclasses import dataclass

import numpy as np
from cares_reinforcement_learning.types.observation import Observation


@dataclass(frozen=True)
class ActionContext:
    observation: Observation
    evaluation: bool
    available_actions: np.ndarray  # TODO redundant with Observation.avail_actions


@dataclass(frozen=True)
class Action:
    action: np.ndarray
