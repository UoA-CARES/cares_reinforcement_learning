import pytest
import numpy as np
from cares_reinforcement_learning.memory import MemoryBuffer  # replace with the correct import path for MemoryBuffer


def test_add_experience():
    buffer = MemoryBuffer(max_capacity=10)
    assert len(buffer) == 0
    buffer.add(action=[1, 2, 3])
    assert len(buffer) == 1
    assert buffer.buffers['action'][0] == [1, 2, 3]


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
    assert len(samples['state']) == 5


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
    assert len(experiences['state']) == 10
    assert len(buffer) == 0


def test_sample_more_than_capacity():
    buffer = MemoryBuffer(max_capacity=10)
    for i in range(5):  # add less experiences than max_capacity
        buffer.add(state=[i,i+1,i+2])
    samples = buffer.sample(10)  # try to sample more experiences than are in the buffer
    assert len(samples['state']) == 5  # it should return all experiences available in the buffer


def test_flush_order():
    buffer = MemoryBuffer(max_capacity=10)
    experiences = []
    for i in range(10):
        experience = [i, i + 1, i + 2]
        buffer.add(state=experience)
        experiences.append(experience)
    flushed_experiences = buffer.flush()
    assert flushed_experiences['state'] == experiences


def test_add_empty_experience():
    buffer = MemoryBuffer(max_capacity=10)
    try:
        buffer.add(**{})  # attempt to add an empty experience
    except Exception:
        pytest.fail("Adding an empty experience raised an exception.")


def test_add_existing_key():
    buffer = MemoryBuffer(max_capacity=10)
    buffer.add(state=[1,2,3])  # add an experience with key 'state'
    buffer.add(state=[4,5,6])  # add another experience with the same key
    assert buffer.buffers['state'][1] == [4,5,6]  # it should overwrite the previous experience


def test_add_when_full():
    buffer = MemoryBuffer(max_capacity=2)
    buffer.add(state=[1,2,3])
    buffer.add(state=[4,5,6])
    assert len(buffer) == 2
    buffer.add(state=[7,8,9])  # add an experience when the buffer is full
    assert len(buffer) == 2  # it should maintain the max_capacity
    assert buffer.buffers['state'][0] == [7,8,9]  # it should overwrite the oldest experience
