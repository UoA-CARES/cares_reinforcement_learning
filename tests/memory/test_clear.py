from memory import memory_buffer


def test_clear(memory_buffer):
    for i in range(5):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    memory_buffer.clear()

    assert len(memory_buffer) == 0
