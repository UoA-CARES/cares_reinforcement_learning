import timeit

import pytest

from cares_reinforcement_learning.memory.memory_buffer import (
    MARLMemoryBuffer,
    SARLMemoryBuffer,
)
from tests.memory.helpers import (
    create_marl_experience,
    create_sarl_experience,
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

    for _ in range(int(1e6) - 1):
        experience = exp_builder()

        buffer.add(experience)

    last_insertion = timeit.timeit(lambda: buffer.add(experience), number=1)

    assert last_insertion < 3 * first_insertion
