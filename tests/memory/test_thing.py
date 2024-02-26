import pytest
from cares_reinforcement_learning.memory import MemoryBuffer

# @pytest.fixture
# def memory_buffer(max_capacity=5):
#     return MemoryBuffer(max_capacity=max_capacity)

# @pytest.mark.parametrize('max_capacity', [1e6])
# def test_consistant_insertion_time(memory_buffer):
#     assert memory_buffer.max_capacity == 1e6