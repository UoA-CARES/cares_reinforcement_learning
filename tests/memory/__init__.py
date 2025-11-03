import pytest
from cares_reinforcement_learning.memory import MemoryBuffer


@pytest.fixture
def memory_buffer():
    return MemoryBuffer(max_capacity=5)


@pytest.fixture
def memory_buffer_1e6():
    return MemoryBuffer(max_capacity=1000000)


@pytest.fixture
def memory_buffer_n_step():
    return MemoryBuffer(max_capacity=1000000, gamma=0.99, n_step=3)
