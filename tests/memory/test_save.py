import os
from pathlib import Path

import numpy as np
import torch
from memory import memory_buffer, memory_buffer_n_step

from cares_reinforcement_learning.memory import MemoryBuffer


def _images_the_same(image_one, image_two):
    if image_one is None and image_two is None:
        return True

    return image_one.shape == image_two.shape and not (
        np.bitwise_xor(image_one, image_two).any()
    )


def _compare_buffer(memory, loaded_memory, experience_size, image_state=False):

    assert len(memory) == len(loaded_memory)

    for i in range(experience_size):
        a = memory.memory_buffers[i]
        b = loaded_memory.memory_buffers[i]
        if i == 0 and image_state:
            for image_a, image_b in zip(a, b):
                assert _images_the_same(image_a, image_b)
        else:
            assert np.array_equal(a, b)

    assert memory.max_capacity == loaded_memory.max_capacity

    assert memory.current_size == loaded_memory.current_size

    assert memory.tree_pointer == loaded_memory.tree_pointer

    assert memory.init_beta == loaded_memory.init_beta

    assert memory.beta == loaded_memory.beta

    assert memory.d_beta == loaded_memory.d_beta

    assert memory.min_priority == loaded_memory.min_priority

    assert memory.max_priority == loaded_memory.max_priority

    sum_tree_levels = memory.sum_tree.levels
    loaded_sum_tree_levels = loaded_memory.sum_tree.levels

    assert len(sum_tree_levels) == len(loaded_sum_tree_levels)

    for i, _ in enumerate(sum_tree_levels):
        a = sum_tree_levels[i]
        b = loaded_sum_tree_levels[i]
        assert np.array_equal(a, b)

    inverse_tree_levels = memory.inverse_tree.levels
    loaded_inverse_tree_levels = loaded_memory.inverse_tree.levels

    assert len(inverse_tree_levels) == len(loaded_inverse_tree_levels)

    for i, _ in enumerate(inverse_tree_levels):
        a = inverse_tree_levels[i]
        b = loaded_inverse_tree_levels[i]
        assert np.array_equal(a, b)


def test_save_load_image(memory_buffer_n_step):
    data_size = 10

    observation_size = (3, 84, 84)

    experience = []
    for i in range(data_size):
        test_image = np.random.randint(0, 255, size=observation_size)
        experience = [test_image, i, i, i, i % 2, i]
        memory_buffer_n_step.add(*experience)

    home = Path.home()

    file_path = f"{home}/cares_rl_logs/test"
    if not os.path.exists(f"{file_path}"):
        os.makedirs(f"{file_path}")

    memory_buffer_n_step.save(file_path, "memory_buffer")

    loaded_memory = MemoryBuffer.load(file_path, "memory_buffer")

    _compare_buffer(
        memory_buffer_n_step, loaded_memory, len(experience), image_state=True
    )


def test_save_load_vector(memory_buffer_n_step):
    data_size = 1000000

    experience = []
    for i in range(data_size):
        experience = [i, i, i, i, i % 2, i]
        memory_buffer_n_step.add(*experience)

    home = Path.home()

    file_path = f"{home}/cares_rl_logs/test"
    if not os.path.exists(f"{file_path}"):
        os.makedirs(f"{file_path}")

    memory_buffer_n_step.save(file_path, "memory_buffer")

    loaded_memory = MemoryBuffer.load(file_path, "memory_buffer")

    _compare_buffer(memory_buffer_n_step, loaded_memory, len(experience))
