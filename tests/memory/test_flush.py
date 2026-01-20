from memory import memory_buffer


def test_flush(memory_buffer):
    for i in range(2):
        memory_buffer.add(i, i, i, i, False)

    sample = memory_buffer.flush()

    assert (
        len(sample.states)
        == len(sample.actions)
        == len(sample.rewards)
        == len(sample.next_states)
        == len(sample.dones)
        == 2
    )
    assert len(memory_buffer) == 0


def test_flush_empty_buffer(memory_buffer):
    sample = memory_buffer.flush()

    assert len(memory_buffer) == 0


def test_flush_order(memory_buffer):
    for i in range(5):
        memory_buffer.add(i, i, i, i, False)

    sample = memory_buffer.flush()

    assert sample.states == [0, 1, 2, 3, 4]
    assert sample.actions == [0, 1, 2, 3, 4]
    assert sample.rewards == [0, 1, 2, 3, 4]
    assert sample.next_states == [0, 1, 2, 3, 4]
    assert sample.dones == [False, False, False, False, False]
