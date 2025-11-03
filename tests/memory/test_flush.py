from memory import memory_buffer


def test_flush(memory_buffer):
    for i in range(2):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    states, actions, rewards, next_states, dones, log_probs = memory_buffer.flush()

    assert (
        len(states)
        == len(actions)
        == len(rewards)
        == len(next_states)
        == len(dones)
        == len(log_probs)
        == 2
    )
    assert len(memory_buffer) == 0


def test_flush_empty_buffer(memory_buffer):
    nothing = memory_buffer.flush()

    assert len(nothing) == 0
    assert len(memory_buffer) == 0


def test_flush_order(memory_buffer):
    for i in range(5):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    states, actions, rewards, next_states, dones, log_probs = memory_buffer.flush()

    assert states == [0, 1, 2, 3, 4]
    assert actions == [0, 1, 2, 3, 4]
    assert rewards == [0, 1, 2, 3, 4]
    assert next_states == [0, 1, 2, 3, 4]
    assert dones == [False, False, False, False, False]
    assert log_probs == [0.0, 0.5, 1.0, 1.5, 2.0]
