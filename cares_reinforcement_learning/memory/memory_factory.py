# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import EpisodicReplayBuffer


class MemoryFactory:
    def create_memory(
        self, buffer_size: int, **priority_params
    ) -> EpisodicReplayBuffer:
        return EpisodicReplayBuffer(
            max_capacity=buffer_size,
            priority_params=priority_params,
        )
