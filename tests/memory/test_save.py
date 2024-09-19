import os
from pathlib import Path

import numpy as np
from memory import memory_buffer, memory_buffer_1e6

from cares_reinforcement_learning.memory import MemoryBuffer


def test_sample_values(memory_buffer_1e6):
    data_size = 10

    experience = []
    for i in range(data_size):
        experience = [i, i, i, i, i % 2, i]
        memory_buffer_1e6.add(*experience)

    home = Path.home()

    file_path = f"{home}/cares_rl_logs/test"
    if not os.path.exists(f"{file_path}"):
        os.makedirs(f"{file_path}")

    memory_buffer_1e6.save(file_path)

    loaded_memory = MemoryBuffer.load(file_path)

    assert len(memory_buffer_1e6) == len(loaded_memory)

    for i in range(len(experience)):
        a = memory_buffer_1e6.memory_buffers[i]
        b = loaded_memory.memory_buffers[i]
        assert np.array_equal(a, b)

    assert memory_buffer_1e6.max_capacity == loaded_memory.max_capacity

    assert memory_buffer_1e6.current_size == loaded_memory.current_size

    assert memory_buffer_1e6.tree_pointer == loaded_memory.tree_pointer

    assert memory_buffer_1e6.init_beta == loaded_memory.init_beta

    assert memory_buffer_1e6.beta == loaded_memory.beta

    assert memory_buffer_1e6.d_beta == loaded_memory.d_beta

    assert memory_buffer_1e6.min_priority == loaded_memory.min_priority

    assert memory_buffer_1e6.max_priority == loaded_memory.max_priority

    sum_tree_levels = memory_buffer_1e6.sum_tree.levels
    loaded_sum_tree_levels = loaded_memory.sum_tree.levels

    assert len(sum_tree_levels) == len(loaded_sum_tree_levels)

    for i, _ in enumerate(sum_tree_levels):
        a = sum_tree_levels[i]
        b = loaded_sum_tree_levels[i]
        assert np.array_equal(a, b)

    inverse_tree_levels = memory_buffer_1e6.inverse_tree.levels
    loaded_inverse_tree_levels = loaded_memory.inverse_tree.levels

    assert len(inverse_tree_levels) == len(loaded_inverse_tree_levels)

    for i, _ in enumerate(inverse_tree_levels):
        a = inverse_tree_levels[i]
        b = loaded_inverse_tree_levels[i]
        assert np.array_equal(a, b)
