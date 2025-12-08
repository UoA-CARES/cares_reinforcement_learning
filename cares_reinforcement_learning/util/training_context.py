from dataclasses import dataclass

import numpy as np
import torch

from cares_reinforcement_learning.memory import MemoryBuffer

# TODO rename this base file


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


@dataclass
class ActionContext:
    state: np.ndarray | dict
    evaluation: bool
    available_actions: np.ndarray


@dataclass
class State:
    # Vector Based
    state: np.ndarray | None = None

    # Image Based
    image: np.ndarray | None = None

    # MARL specific
    agent_observations: dict[str, np.ndarray] | None = None
    avail_actions: np.ndarray | None = None


@dataclass
class StateTensors:
    # Vector Based
    states_tensor: torch.Tensor | None = None

    # Image Based
    images_tensor: torch.Tensor | None = None

    # MARL specific
    agent_observations_tensor: dict[str, torch.Tensor] | None = None
    avail_actions_tensor: torch.Tensor | None = None


@dataclass
class Experience:
    state: State
    next_state: State


@dataclass
class SAExperience(Experience):
    reward: float
    done: bool
    truncated: bool

    @property
    def done_flag(self) -> bool:
        return self.done_flag

    @property
    def truncated_flag(self) -> bool:
        return self.truncated_flag

    def reward_sum(self) -> float:
        return self.reward


@dataclass
class MAExperience(Experience):
    reward: dict[str, float]
    done: dict[str, bool]
    truncated: dict[str, bool]

    @property
    def done_flag(self) -> bool:
        # terminal when *all* agents are done
        return all(self.done.values())

    @property
    def truncated_flag(self) -> bool:
        # partial truncation shouldn't terminate the episode from training-loop POV
        return all(self.truncated.values())

    def reward_sum(self) -> float:
        return float(sum(self.reward.values()))
