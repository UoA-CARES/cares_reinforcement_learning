import random
from collections import deque


class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):
        self.buffer = deque(maxlen=max_capacity)

    def __len__(self):
        return len(self.buffer)

    def add(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        experience_batch = random.sample(self.buffer, batch_size)
        transposed_batch = zip(*experience_batch)
        return transposed_batch

    def flush(self):
        experience = list(zip(*self.buffer))
        self.buffer.clear()
        return experience

    def clear(self):
        self.buffer.clear()