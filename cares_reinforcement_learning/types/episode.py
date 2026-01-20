from dataclasses import dataclass


@dataclass
class EpisodeContext:
    training_step: int

    # Episode-specific context
    episode: int
    episode_steps: int
    episode_reward: float
    episode_done: bool
