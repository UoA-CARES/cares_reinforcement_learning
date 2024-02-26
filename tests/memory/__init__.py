import pytest
from cares_reinforcement_learning.memory import MemoryBuffer

@pytest.fixture
def memory_buffer():
    return MemoryBuffer(max_capacity=5)

@pytest.fixture
def memory_buffer_1e6():
    return MemoryBuffer(max_capacity=1000000)