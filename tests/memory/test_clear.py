import pytest

from cares_reinforcement_learning.memory.memory_buffer import (
    SARLMemoryBuffer,
    MARLMemoryBuffer,
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
def test_clear(buffer_class, exp_builder):
    """Test that clearing the buffer removes all experiences."""
    buffer = buffer_class(max_capacity=5)

    for i in range(5):
        experience = exp_builder()
        buffer.add(experience)

    buffer.clear()

    assert len(buffer) == 0
