# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

import cares_reinforcement_learning.memory.priority_functions as pf
from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


def get_priority_function(memory_type):
    if memory_type == "MemoryBuffer":
        return pf.standard
    elif memory_type == "PER":
        return pf.td_error
    elif memory_type == "algorithm":
        return pf.algorithm_priority
    raise ValueError(f"Unkown memory type: {memory_type}")


class MemoryFactory:
    def create_memory(self, memory_type, buffer_size, **priority_params):

        priority_function = get_priority_function(memory_type)

        return PrioritizedReplayBuffer(
            max_size=buffer_size,
            priority_function=priority_function,
            priority_params=priority_params,
        )
