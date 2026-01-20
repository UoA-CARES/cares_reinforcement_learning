import numpy as np
from memory import memory_buffer, memory_buffer_1e6, memory_buffer_n_step


def test_sample(memory_buffer_1e6):

    for i in range(4):
        memory_buffer_1e6.add(i, i, i, i, False)

    batch_size = 3
    sample = memory_buffer_1e6.sample_uniform(batch_size)

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == len(sample.indices)
        == batch_size
    )

    sample = memory_buffer_1e6.sample_inverse_priority(batch_size)

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == len(sample.indices)
        == len(sample.weights)
        == batch_size
    )

    sample = memory_buffer_1e6.sample_priority(batch_size)

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == len(sample.indices)
        == len(sample.weights)
        == batch_size
    )


def test_n_step_values(memory_buffer_n_step):
    memory_buffer_n_step.n_step = 3  # Get n-step value
    memory_buffer_n_step.gamma = 0.99  # Discount factor

    # Add transitions: [state, action, reward, next_state, done]
    memory_buffer_n_step.add(1, 1, 1, 1, 0)
    memory_buffer_n_step.add(2, 2, 2, 2, 0)
    memory_buffer_n_step.add(3, 3, 3, 3, 0)

    # Sample a batch of size 1
    sample = memory_buffer_n_step.sample_uniform(1)

    value = sample.indices[0]  # Get sampled index

    # Compute expected n-step return
    expected_reward = 5.9203

    # Assertions to validate n-step transitions
    assert sample.states == [value + 1]
    assert sample.actions == [value + 1]
    assert np.isclose(sample.rewards[0], expected_reward)  # Check discounted sum
    assert sample.next_states == [3]  # Expected next state
    assert sample.dones == [0]  # Done flag based on sequence
    assert sample.indices == [value]


def test_sample_values(memory_buffer_1e6):
    for i in range(10):
        memory_buffer_1e6.add(i, i, i, i, i % 2)

    sample = memory_buffer_1e6.sample_uniform(1)

    value = sample.indices[0]
    assert sample.states == [value]
    assert sample.actions == [value]
    assert sample.rewards == [value]
    assert sample.next_states == [value]
    assert sample.dones == [value % 2]
    assert sample.indices == [value]


def test_sample_priority_values(memory_buffer_1e6):
    for i in range(10):
        memory_buffer_1e6.add(i, i, i, i, i % 2)

    sample = memory_buffer_1e6.sample_priority(1)

    value = sample.indices[0]
    assert sample.states == [value]
    assert sample.actions == [value]
    assert sample.rewards == [value]
    assert sample.next_states == [value]
    assert sample.dones == [value % 2]
    assert sample.indices == [value]
    assert sample.weights == [1.0]


def test_sample_inverse_sample_values(memory_buffer_1e6):
    size = 10
    for i in range(size):
        memory_buffer_1e6.add(i, i, i, i, i % 2)

    ind = []
    priorities = []
    for i in range(size):
        ind.append(i)
        priorities.append(1)
    memory_buffer_1e6.update_priorities(np.array(ind), np.array(priorities))

    sample = memory_buffer_1e6.sample_inverse_priority(1)

    value = sample.indices[0]
    assert sample.states == [value]
    assert sample.actions == [value]
    assert sample.rewards == [value]
    assert sample.next_states == [value]
    assert sample.dones == [value % 2]
    assert sample.indices == [value]
    assert abs(sample.weights[0] - size) < 0.001


def test_sample_consecutive_values(memory_buffer_1e6):
    for i in range(20):
        memory_buffer_1e6.add(i, i, i, i, i % 2)

    sample_one, sample_two = memory_buffer_1e6.sample_consecutive(1)

    value = sample_one.indices[0]
    assert sample_one.states == [value]
    assert sample_one.actions == [value]
    assert sample_one.rewards == [value]
    assert sample_one.next_states == [value]
    assert sample_one.dones == [False]

    assert sample_two.states == [value + 1]
    assert sample_two.actions == [value + 1]
    assert sample_two.rewards == [value + 1]
    assert sample_two.next_states == [value + 1]
    assert sample_two.dones == [True]

    assert sample_one.indices == [value]


def test_sample_more_than_buffer(memory_buffer):
    for i in range(5):
        memory_buffer.add(i, i, i, i, False)

    batch_size = 10
    sample = memory_buffer.sample_uniform(batch_size)

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == len(sample.indices)
        == len(sample.weights)
        == 5
    )

    sample = memory_buffer.sample_priority(batch_size)

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == len(sample.indices)
        == len(sample.weights)
        == 5
    )

    sample = memory_buffer.sample_inverse_priority(batch_size)

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == len(sample.indices)
        == len(sample.weights)
        == 5
    )


def test_sample_empty_buffer(memory_buffer):
    batch_size = 10
    try:
        sample = memory_buffer.sample_uniform(batch_size)
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True

    try:
        sample = memory_buffer.sample_priority(batch_size)
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True

    try:
        sample = memory_buffer.sample_inverse_priority(batch_size)
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True

    try:
        sample = memory_buffer.sample_consecutive(1)
    except ValueError:
        assert True
