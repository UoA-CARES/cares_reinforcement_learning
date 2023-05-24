import pytest
import random 

from cares_reinforcement_learning.memory import MemoryBuffer

@pytest.fixture
def memory_buffer():
    return MemoryBuffer()

def test_add(memory_buffer):
    memory_size = 100
    for _ in range(0, memory_size):
        memory_buffer.add(state=[0,0,0,0], action=[1], reward=0, next_state=[1,1,1,1], done=False)
    assert len(memory_buffer) == memory_size, f"buffer has not had the experience added correctly - current size {len(memory_buffer)}"

@pytest.fixture
def filled_memory_buffer():
    memory_buffer = MemoryBuffer(max_capacity=100)
    for _ in range(0, 100):
        state = [random.random() for _ in range(4)]
        action = [random.randint(0, 1)]
        reward = random.random()
        next_state = [random.random() for _ in range(4)]
        done = random.choice([True, False])
        memory_buffer.add(state=state, action=action, reward=reward, next_state=next_state, done=done)
    return memory_buffer

def test_add_full_buffer(filled_memory_buffer):
    assert len(filled_memory_buffer) == 100, f"buffer has not had the experience added correctly - current size {len(filled_memory_buffer)}"