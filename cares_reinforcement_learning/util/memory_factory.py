# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *


class MemoryFactory:
    def create_memory(self, memory_type, buffer_size, arg):
        if memory_type == "MemoryBuffer":
            return MemoryBuffer(max_capacity=buffer_size)
        if memory_type == "PER":
            return MemoryBuffer(augment=td_error)
        raise ValueError(f"Unkown memory type: {memory_type}")
