import numpy as np
from memory import memory_buffer, memory_buffer_1e6, memory_buffer_n_step


def test_sample(memory_buffer_1e6):

    for i in range(4):
        memory_buffer_1e6.add(i, i, i, i, False, 0.5 * i)

    batch_size = 3
    states, actions, rewards, next_states, dones, log_probs, ind = (
        memory_buffer_1e6.sample_uniform(batch_size)
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == len(ind)
        == batch_size
    )

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer_1e6.sample_inverse_priority(batch_size)
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == len(ind)
        == len(weights)
        == batch_size
    )

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer_1e6.sample_priority(batch_size)
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == len(ind)
        == len(weights)
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
    states, actions, rewards, next_states, dones, ind = (
        memory_buffer_n_step.sample_uniform(1)
    )

    value = ind[0]  # Get sampled index

    # Compute expected n-step return
    expected_reward = 5.9203

    # Assertions to validate n-step transitions
    assert states == [value + 1]
    assert actions == [value + 1]
    assert np.isclose(rewards[0], expected_reward)  # Check discounted sum
    assert next_states == [3]  # Expected next state
    assert dones == [0]  # Done flag based on sequence
    assert ind == [value]


def test_sample_values(memory_buffer_1e6):
    for i in range(10):
        memory_buffer_1e6.add(i, i, i, i, i % 2, i)

    states, actions, rewards, next_states, dones, log_probs, ind = (
        memory_buffer_1e6.sample_uniform(1)
    )

    value = ind[0]
    assert states == [value]
    assert actions == [value]
    assert rewards == [value]
    assert next_states == [value]
    assert dones == [value % 2]
    assert log_probs == [value]
    assert ind == [value]


def test_sample_priority_values(memory_buffer_1e6):
    for i in range(10):
        memory_buffer_1e6.add(i, i, i, i, i % 2, i)

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer_1e6.sample_priority(1)
    )

    value = ind[0]
    assert states == [value]
    assert actions == [value]
    assert rewards == [value]
    assert next_states == [value]
    assert dones == [value % 2]
    assert log_probs == [value]
    assert ind == [value]
    assert weights == [1.0]


def test_sample_inverse_sample_values(memory_buffer_1e6):
    size = 10
    for i in range(size):
        memory_buffer_1e6.add(i, i, i, i, i % 2, i)

    ind = []
    priorities = []
    for i in range(size):
        ind.append(i)
        priorities.append(1)
    memory_buffer_1e6.update_priorities(np.array(ind), np.array(priorities))

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer_1e6.sample_inverse_priority(1)
    )

    value = ind[0]
    assert states == [value]
    assert actions == [value]
    assert rewards == [value]
    assert next_states == [value]
    assert dones == [value % 2]
    assert log_probs == [value]
    assert ind == [value]
    assert abs(weights[0] - size) < 0.001


def test_sample_consecutive_values(memory_buffer_1e6):
    for i in range(20):
        memory_buffer_1e6.add(i, i, i, i, i % 2)

    (
        states_t1,
        actions_t1,
        rewards_t1,
        next_states_t1,
        dones_t1,
        states_t2,
        actions_t2,
        rewards_t2,
        next_states_t2,
        dones_t2,
        ind,
    ) = memory_buffer_1e6.sample_consecutive(1)

    value = ind[0]
    assert states_t1 == [value]
    assert actions_t1 == [value]
    assert rewards_t1 == [value]
    assert next_states_t1 == [value]
    assert dones_t1 == [False]

    assert states_t2 == [value + 1]
    assert actions_t2 == [value + 1]
    assert rewards_t2 == [value + 1]
    assert next_states_t2 == [value + 1]
    assert dones_t2 == [True]

    assert ind == [value]


def test_sample_more_than_buffer(memory_buffer):
    for i in range(5):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    batch_size = 10
    states, actions, rewards, next_states, dones, log_probs, ind = (
        memory_buffer.sample_uniform(batch_size)
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == len(ind)
        == 5
    )

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer.sample_priority(batch_size)
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == len(ind)
        == len(weights)
        == 5
    )

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer.sample_inverse_priority(batch_size)
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == len(ind)
        == len(weights)
        == 5
    )


def test_sample_empty_buffer(memory_buffer):
    batch_size = 10
    try:
        states, actions, rewards, next_states, dones, indicies = (
            memory_buffer.sample_uniform(batch_size)
        )
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True

    try:
        states, actions, rewards, next_states, dones, indicies, weights = (
            memory_buffer.sample_priority(batch_size)
        )
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True

    try:
        states, actions, rewards, next_states, dones, indicies, weights = (
            memory_buffer.sample_inverse_priority(batch_size)
        )
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True

    try:
        (
            states_t1,
            actions_t1,
            rewards_t1,
            next_states_t1,
            dones_t1,
            states_t2,
            actions_t2,
            rewards_t2,
            next_states_t2,
            dones_t2,
            ind,
        ) = memory_buffer.sample_consecutive(1)
    except ValueError:
        assert True
