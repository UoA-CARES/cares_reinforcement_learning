# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *


class MemoryFactory:
    def create_memory(self, memory_type, state_dim, action_dim):
        if memory_type == "MemoryBuffer":
            return MemoryBuffer(state_dim=state_dim, action_dim=action_dim)

        if memory_type == "PER":
            return MemoryBuffer()
        raise ValueError(f"Unkown memory type: {memory_type}")
