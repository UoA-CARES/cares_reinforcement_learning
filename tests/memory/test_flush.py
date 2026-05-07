import pytest
import numpy as np

from cares_reinforcement_learning.memory.memory_buffer import (
    SARLMemoryBuffer,
    MARLMemoryBuffer,
)
from tests.memory.helpers import (
    create_marl_experience,
    create_sarl_experience,
    get_indexed_marl_observation,
    get_indexed_sarl_observation,
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
        if buffer_class is SARLMemoryBuffer:
            obs = get_indexed_sarl_observation(i)
            next_obs = get_indexed_sarl_observation(i)
        else:
            obs = get_indexed_marl_observation(i)
            next_obs = get_indexed_marl_observation(i)

        experience = exp_builder(observation=obs, next_observation=next_obs)
        buffer.add(experience)

    sample = buffer.flush()

    # Verify we have the correct number of experiences
    assert len(sample.experiences) == 5
    assert len(sample.indices) == 5
    assert len(sample.weights) == 5
    assert sample.indices == list(range(5))

    # Verify experiences preserve insertion order
    for i, exp in enumerate(sample.experiences):
        if buffer_class is SARLMemoryBuffer:
            assert np.allclose(exp.observation.vector_state, float(i))
            assert np.allclose(exp.next_observation.vector_state, float(i))
        else:
            assert np.allclose(exp.observation.global_state, float(i))
            assert np.allclose(exp.next_observation.global_state, float(i))
    assert len(buffer) == 0
