from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional

from cares_reinforcement_learning.memory import MemoryBuffer


class TrainingEvent(Enum):
    STEP = "step"
    EPISODE_END = "episode_end"


@dataclass
class TrainingContext:
    # Core training parameters
    memory: MemoryBuffer
    batch_size: int  # can just differ this to the algorithm itself
    event: TrainingEvent
    training_step: int

    # Episode-specific context
    episode: int
    episode_steps: int
    episode_reward: float

    # Environment context (optional)
    env_info: Optional[Dict[str, Any]] = None

    # Training state context (optional)
    exploration_phase: bool = False
