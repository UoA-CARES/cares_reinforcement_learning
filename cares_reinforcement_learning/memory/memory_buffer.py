import random
from collections import deque


class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        experience_batch = random.sample(self.buffer, batch_size)
        transposed_batch = zip(*experience_batch)
        return transposed_batch

    def flush(self):
        states, actions, rewards, next_states, dones, log_probs = zip(
            *[(element[i] for i in range(len(element))) for element in self.buffer]
        )
        self.buffer.clear()
        return states, actions, rewards, next_states, dones, log_probs
