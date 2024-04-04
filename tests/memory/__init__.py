import pytest
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


@pytest.fixture
def memory_buffer():
    return PrioritizedReplayBuffer(max_capacity=5)


@pytest.fixture
def memory_buffer_1e6():
    return PrioritizedReplayBuffer(max_capacity=1000000)
