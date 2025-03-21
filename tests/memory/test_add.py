import timeit
from memory import memory_buffer, memory_buffer_n_step


def test_add(memory_buffer):
    memory_buffer.add(1, 2, 3, 4, False, 0.5)
    assert len(memory_buffer) == 1


def test_buffer_full(memory_buffer):

    for i in range(5):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    assert len(memory_buffer) == 5


def test_overfill_buffer(memory_buffer):
    for i in range(6):
        memory_buffer.add(i, i, i, i, False, 0.5 * i)

    assert len(memory_buffer) == 5


def test_consistant_insertion_time(memory_buffer_n_step):

    first_insertion = timeit.timeit(
        lambda: memory_buffer_n_step.add(1, 2, 3, 4, False, 0.5), number=1
    )

    for i in range(1000000):
        memory_buffer_n_step.add(i, i, i, i, False, 0.5 * i)

    assert len(memory_buffer_n_step) == 1e6

    last_insertion = timeit.timeit(
        lambda: memory_buffer_n_step.add(1, 2, 3, 4, False, 0.5), number=1
    )

    assert last_insertion < 10 * first_insertion
