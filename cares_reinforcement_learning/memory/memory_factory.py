# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory.episodic_replay_buffer import ManageBuffers
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


class MemoryFactory:
    def create_memory(
        self, alg_config: AlgorithmConfig) -> ManageBuffers:
        return ManageBuffers(
            max_capacity=alg_config.buffer_size,
            priority_params={},
        )
