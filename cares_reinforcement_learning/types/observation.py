from dataclasses import dataclass

import numpy as np
import torch


# TODO split into SingleAgentObservation and MultiAgentObservation?
@dataclass
class Observation:
    # Vector Based
    vector_state: np.ndarray

    # Image Based
    image_state: np.ndarray | None = None

    # MARL specific
    agent_states: dict[str, np.ndarray] | None = None
    avail_actions: np.ndarray | None = None


@dataclass
class ObservationTensors:
    # Vector Based
    vector_state_tensor: torch.Tensor

    # Image Based
    image_state_tensor: torch.Tensor | None = None

    # MARL specific
    agent_states_tensor: dict[str, torch.Tensor] | None = None
    avail_actions_tensor: torch.Tensor | None = None
