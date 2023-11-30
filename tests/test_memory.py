import pytest
import numpy as np
from cares_reinforcement_learning.memory import MemoryBuffer
from collections import deque


def test_add_experience():
    buffer = MemoryBuffer(max_capacity=10)
    assert len(buffer) == 0
    buffer.add(action=[1, 2, 3])
    assert len(buffer) == 1
    assert buffer.buffers["action"][0] == [1, 2, 3]


def test_buffer_full():
    buffer = MemoryBuffer(max_capacity=2)
    buffer.add(state=[1, 2, 3])
    assert buffer.full == False
    buffer.add(state=[7, 8, 9])
    assert buffer.full == True


def test_sample_experience():
    buffer = MemoryBuffer(max_capacity=10)
    for i in range(10):
        buffer.add(state=[i, i + 1, i + 2])
    samples = buffer.sample(5)
    assert len(samples["state"]) == 5


def test_clear_buffer():
    buffer = MemoryBuffer(max_capacity=10)
    for i in range(10):
        buffer.add(state=[i, i + 1, i + 2])
    buffer.clear()
    assert len(buffer) == 0


def test_flush_buffer():
    buffer = MemoryBuffer(max_capacity=10)
    for i in range(10):
        buffer.add(state=[i, i + 1, i + 2])
    experiences = buffer.flush()
    assert len(experiences["state"]) == 10
    assert len(buffer) == 0


def test_sample_more_than_capacity():
    buffer = MemoryBuffer(max_capacity=10)
    for i in range(5):
        buffer.add(state=[i, i + 1, i + 2])
    samples = buffer.sample(10)
    assert len(samples["state"]) == 5


def test_flush_order():
    buffer = MemoryBuffer(max_capacity=10)
    experiences = []
    for i in range(10):
        experience = [i, i + 1, i + 2]
        buffer.add(state=experience)
        experiences.append(experience)
    flushed_experiences = buffer.flush()
    assert flushed_experiences["state"] == experiences


def test_add_empty_experience():
    buffer = MemoryBuffer(max_capacity=10)
    try:
        buffer.add(**{})
    except Exception:
        pytest.fail("Adding an empty experience raised an exception.")


def test_add_existing_key():
    buffer = MemoryBuffer(max_capacity=10)
    buffer.add(state=[1, 2, 3])
    buffer.add(state=[4, 5, 6])
    assert buffer.buffers["state"][1] == [4, 5, 6]


def test_add_when_full():
    buffer = MemoryBuffer(max_capacity=2)
    buffer.add(state=[1, 2, 3])
    buffer.add(state=[4, 5, 6])
    assert len(buffer) == 2
    buffer.add(state=[7, 8, 9])
    assert len(buffer) == 2
    assert buffer.buffers["state"][0] == [7, 8, 9]


class TestMemoryBuffer:
    def setup_method(self):
        self.buffer = MemoryBuffer(max_capacity=100, eps=0.01, alpha=0.6)
        self.buffer.add(test_data1=1, test_data2=2, test_data3=3)
        self.buffer.add(test_data1=1, test_data2=2, test_data3=3)
        self.buffer.add(test_data1=1, test_data2=2, test_data3=3)

    def test_update_priorities(self):
        indices = [0, 1, 2]
        info = {"test_info": 0.5}

        def mock_augment(indices, info, params):
            return np.array([1.0, 2.0, 3.0])

        self.buffer.augment = mock_augment
        self.buffer.update_priorities(indices, info)

    def test_update_priorities_with_nonexistent_index(self):
        indices = [0, 1, 10]  # 10 is nonexistent
        info = {"test_info": 0.5}

        def mock_augment(indices, info, params):
            return np.array([1.0, 2.0, 3.0])

        self.buffer.augment = mock_augment
        with pytest.raises(IndexError):
            self.buffer.update_priorities(indices, info)
