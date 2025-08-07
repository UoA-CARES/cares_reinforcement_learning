# Intentionally import all augmentations
# pylint: disable=wildcard-import, unused-wildcard-import

from cares_reinforcement_learning.memory import PrioritizedReplayBuffer
from cares_reinforcement_learning.util.configurations import AlgorithmConfig
from cares_reinforcement_learning.memory.episodic_replay_buffer import ManageBuffers


class MemoryFactory:
    def create_memory(self, alg_config: AlgorithmConfig):

        beta = 0.0
        d_beta = 0.0
        min_priority = 1.0
        buffer_size = alg_config.buffer_size

        if hasattr(alg_config, "beta"):
            beta = alg_config.beta
            d_beta = (1.0 - alg_config.beta) / alg_config.max_steps_training

        if hasattr(alg_config, "min_priority"):
            min_priority = alg_config.min_priority
        # print(alg_config)
        # input()
        if alg_config.algorithm in ["EpisodicTD3", "ReTD3SIL", "ReTD3SIL", "RESAC", "ReSurpriseTD3", "RePPO"]:
         
            return ManageBuffers(
                max_capacity=alg_config.buffer_size,
                priority_params={},
            )

        return PrioritizedReplayBuffer(
            max_capacity=buffer_size,
            min_priority=min_priority,
            beta=beta,
            d_beta=d_beta,
            priority_params={},
        )
