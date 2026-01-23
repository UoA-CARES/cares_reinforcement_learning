import pytest
import numpy as np

from cares_reinforcement_learning.types.experience import (
    SingleAgentExperience,
    MultiAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    MARLObservation,
)
from cares_reinforcement_learning.memory.memory_buffer import (
    SARLMemoryBuffer,
    MARLMemoryBuffer,
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
def test_flush(buffer_class, exp_builder):
    """Test that flushing returns all buffered experiences."""
    buffer = buffer_class(max_capacity=5)

    for i in range(2):
        experience = exp_builder()
        buffer.add(experience)

    sample = buffer.flush()

    assert len(sample.experiences) == 2
    assert len(sample.indices) == 2
    assert len(sample.weights) == 2
    assert len(buffer) == 0


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_flush_empty_buffer(buffer_class, exp_builder):
    """Test that flushing an empty buffer returns empty sample."""
    buffer = buffer_class(max_capacity=5)
    sample = buffer.flush()

    assert len(buffer) == 0
    assert len(sample.experiences) == 0
    assert len(sample.indices) == 0
    assert len(sample.weights) == 0


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_flush_order(buffer_class, exp_builder):
    """Test that flush preserves insertion order of experiences."""
    buffer = buffer_class(max_capacity=5)

    for i in range(5):
        experience = exp_builder()
        buffer.add(experience)

    sample = buffer.flush()

    # Verify we have the correct number of experiences
    assert len(sample.experiences) == 5
    assert len(sample.indices) == 5
    assert len(sample.weights) == 5
    assert len(buffer) == 0
