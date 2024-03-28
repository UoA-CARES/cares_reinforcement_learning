# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.memory.augments import *

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


class MemoryFactory:
    def create_memory(self, memory_type, buffer_size, **kwargs):
        if memory_type == "MemoryBuffer":
            return MemoryBuffer(max_capacity=buffer_size)
        if memory_type == "PER":
            if "observation_size" not in kwargs:
                raise AttributeError(
                    "observation_size not passed into kwargs. PER memory buffer is dependent on this."
                )

            if "action_num" not in kwargs:
                raise AttributeError(
                    "action_num not passed into kwargs. PER memory buffer is dependent on this."
                )

            return PrioritizedReplayBuffer(
                kwargs["observation_size"], kwargs["action_num"], max_size=buffer_size
            )
        raise ValueError(f"Unkown memory type: {memory_type}")
