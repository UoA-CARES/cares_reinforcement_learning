from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

import numpy as np

ActType = TypeVar("ActType", bound=int | np.ndarray | list[int] | list[np.ndarray])


@dataclass(frozen=True, slots=True)
class ActionSample(Generic[ActType]):
    action: ActType
    source: str  # e.g., "policy", "exploration/random", "expert"
    extras: dict[str, Any] = field(default_factory=dict)
