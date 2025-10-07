from dataclasses import dataclass

from cares_reinforcement_learning.memory import MemoryBuffer


@dataclass
class TrainingContext:
    # Core training parameters
    memory: MemoryBuffer
    batch_size: int  # can just differ this to the algorithm itself
    training_step: int

    # Episode-specific context
    episode: int
    episode_steps: int
    episode_reward: float
    episode_done: bool
