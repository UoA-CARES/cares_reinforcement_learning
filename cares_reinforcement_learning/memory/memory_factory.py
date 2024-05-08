# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
from cares_reinforcement_learning.util.configurations import AlgorithmConfig


class MemoryFactory:
    def create_memory(self, alg_config: AlgorithmConfig) -> PrioritizedReplayBuffer:

        beta = 0.0
        d_beta = 0.0
        min_priority = 1.0

        if hasattr(alg_config, "beta"):
            beta = alg_config.beta
            d_beta = (1.0 - alg_config.beta) / alg_config.max_steps_training

        if hasattr(alg_config, "min_priority"):
            min_priority = alg_config.min_priority

        return PrioritizedReplayBuffer(
            max_capacity=alg_config.buffer_size,
            min_priority=min_priority,
            beta=beta,
            d_beta=d_beta,
            priority_params={},
        )
