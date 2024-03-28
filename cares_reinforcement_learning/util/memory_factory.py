# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import MemoryBuffer, MemoryBufferMBRL
from cares_reinforcement_learning.memory.augments import *


class MemoryFactory:
    def create_memory(self, memory_type, buffer_size, args):
        if memory_type == "MemoryBuffer":
            return MemoryBuffer(max_capacity=buffer_size)
        if memory_type == "PER":
            return MemoryBuffer(augment=td_error)
        if memory_type == "MemoryBufferMBRL":
            return MemoryBufferMBRL(max_capacity=buffer_size)
        raise ValueError(f"Unkown memory type: {memory_type}")
