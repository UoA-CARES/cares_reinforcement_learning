from memory import memory_buffer, memory_buffer_1e6


def test_sample(memory_buffer):

    for i in range(4):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    batch_size = 3
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
        == batch_size
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
        == batch_size
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
        == batch_size
    )


def test_sample_values(memory_buffer):
    memory_buffer.add(0.5, 1.0, 1.5, 2.0, False, 0.5)

    states, actions, rewards, next_states, dones, log_probs, ind = (
        memory_buffer.sample_uniform(1)
    )

    assert states == [0.5]
    assert actions == [1.0]
    assert rewards == [1.5]
    assert next_states == [2.0]
    assert dones == [False]
    assert log_probs == [0.5]
    assert ind == [0]

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer.sample_priority(1)
    )

    assert states == [0.5]
    assert actions == [1.0]
    assert rewards == [1.5]
    assert next_states == [2.0]
    assert dones == [False]
    assert log_probs == [0.5]
    assert ind == [0]
    assert weights == [1.0]

    states, actions, rewards, next_states, dones, log_probs, ind, weights = (
        memory_buffer.sample_inverse_priority(1)
    )

    assert states == [0.5]
    assert actions == [1.0]
    assert rewards == [1.5]
    assert next_states == [2.0]
    assert dones == [False]
    assert log_probs == [0.5]
    assert ind == [0]
    assert abs(weights[0] - 1.0) < 0.0001


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
