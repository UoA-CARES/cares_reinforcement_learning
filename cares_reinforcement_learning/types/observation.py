from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class SARLObservation:
    # Vector Based
    vector_state: np.ndarray

    # Image Based
    image_state: np.ndarray | None = None

    avail_actions: np.ndarray | None = None


@dataclass
class MARLObservation:

    global_state: np.ndarray
    # Per-Agent States
    agent_states: dict[str, np.ndarray]

    avail_actions: np.ndarray


Observation = SARLObservation | MARLObservation


@dataclass
class SARLObservationTensors:
    # Vector Based
    vector_state_tensor: torch.Tensor

    # Image Based
    image_state_tensor: torch.Tensor | None = None


@dataclass
class MARLObservationTensors:
    # Global State
    global_state_tensor: torch.Tensor

    # Per-Agent States
    agent_states_tensor: dict[str, torch.Tensor]
    avail_actions_tensor: torch.Tensor


ObservationTensors = SARLObservationTensors | MARLObservationTensors
