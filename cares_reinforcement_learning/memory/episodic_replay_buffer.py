

from cares_reinforcement_learning.memory.short_term_replay_buffer import ShortTermReplayBuffer
from cares_reinforcement_learning.memory.long_memory_buffer import LongMemoryBuffer
from cares_reinforcement_learning.memory.episodic_buffer import EpisodicBuffer


class ManageBuffers:
   

    def __init__(self, max_capacity: int = int(1e6), **memory_params):
        self.memory_params = memory_params
        
        self.short_term_memory = ShortTermReplayBuffer(max_capacity=max_capacity, **memory_params)
        self.long_term_memory = LongMemoryBuffer(max_capacity=int(10), **memory_params)
        #self.episodic_memory = EpisodicBuffer(max_capacity=max_capacity)
        
