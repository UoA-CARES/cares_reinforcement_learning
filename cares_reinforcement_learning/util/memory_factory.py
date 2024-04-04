# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer


class MemoryFactory:
    def create_memory(self, buffer_size, **priority_params):
        return PrioritizedReplayBuffer(
            max_capacity=buffer_size,
            priority_params=priority_params,
        )
