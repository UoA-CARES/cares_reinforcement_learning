from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *

class MemoryFactory:
    def create_memory(self, memory_type, args):
        if memory_type == "MemoryBuffer":
            return MemoryBuffer()
        elif memory_type == "PER":
            return MemoryBuffer(augment=td_error)
        else:
            raise ValueError(f"Unkown memory type: {memory_type}")