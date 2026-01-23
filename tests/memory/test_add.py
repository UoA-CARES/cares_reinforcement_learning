import timeit

import numpy as np
import pytest

from cares_reinforcement_learning.memory.memory_buffer import (
    MARLMemoryBuffer,
    SARLMemoryBuffer,
)
from cares_reinforcement_learning.types.experience import (
    MultiAgentExperience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


def get_sarl_observation(state_size: int = 4) -> SARLObservation:
    """Create a dummy SARL observation for testing."""
    return SARLObservation(vector_state=np.array([1.0] * state_size), image_state=None)


def get_marl_observation(state_size: int = 4, num_agents: int = 2) -> MARLObservation:
    """Create a dummy MARL observation for testing."""
    agent_states = {
        f"agent_{i}": np.array([1.0] * state_size) for i in range(num_agents)
    }
    return MARLObservation(
        global_state=np.array([1.0] * state_size),
        agent_states=agent_states,
        avail_actions=np.ones((num_agents,), dtype=bool),
    )


def create_sarl_experience(
    observation: SARLObservation | None = None,
    next_observation: SARLObservation | None = None,
) -> SingleAgentExperience:
    """Create a SingleAgentExperience for testing."""
    if observation is None:
        observation = get_sarl_observation()
    if next_observation is None:
        next_observation = get_sarl_observation()

    return SingleAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=np.array([0.5]),
        reward=1.0,
        done=False,
        truncated=False,
        info={},
    )


def create_marl_experience(
    observation: MARLObservation | None = None,
    next_observation: MARLObservation | None = None,
) -> MultiAgentExperience:
    """Create a MultiAgentExperience for testing."""
    if observation is None:
        observation = get_marl_observation()
    if next_observation is None:
        next_observation = get_marl_observation()

    return MultiAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=[np.array([0.5]), np.array([0.3])],
        reward=[1.0, 0.5],
        done=[False, False],
        truncated=[False, False],
        info={},
    )


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_add(buffer_class, exp_builder):
    """Test that adding a single experience to the buffer works."""
    buffer = buffer_class(max_capacity=5)
    experience = exp_builder()
    buffer.add(experience)
    assert len(buffer) == 1


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_buffer_full(buffer_class, exp_builder):
    """Test that buffer respects max_capacity when adding experiences."""
    buffer = buffer_class(max_capacity=5)

    for i in range(5):
        experience = exp_builder()
        buffer.add(experience)

    assert len(buffer) == 5


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_overfill_buffer(buffer_class, exp_builder):
    """Test that buffer overwrites oldest experience when exceeding capacity."""
    buffer = buffer_class(max_capacity=5)

    for i in range(6):
        experience = exp_builder()
        buffer.add(experience)

    assert len(buffer) == 5


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_consistent_insertion_time(buffer_class, exp_builder):
    """Test that insertion time remains consistent with buffer size."""
    buffer = buffer_class(max_capacity=int(1e6))
    experience = exp_builder()

    first_insertion = timeit.timeit(lambda: buffer.add(experience), number=1)

    for _ in range(1000000):
        experience = exp_builder()

        buffer.add(experience)

    last_insertion = timeit.timeit(lambda: buffer.add(experience), number=1)

    assert last_insertion < 10 * first_insertion
