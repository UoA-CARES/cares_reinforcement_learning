from memory import memory_buffer


def test_sample(memory_buffer):

    for i in range(4):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    batch_size = 3
    states, actions, rewards, next_states, dones, log_probs = memory_buffer.sample(
        batch_size
    )
    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == batch_size
    )


def test_sample_more_than_buffer(memory_buffer):
    for i in range(5):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    batch_size = 10
    states, actions, rewards, next_states, dones, log_probs = memory_buffer.sample(
        batch_size
    )

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == 5
    )


def test_sample_empty_buffer(memory_buffer):
    batch_size = 10
    try:
        states, actions, rewards, next_states, dones, log_probs = memory_buffer.sample(
            batch_size
        )
    except ValueError:
        # Nothing in the buffer, so should receive not enough values to unpack
        assert True
