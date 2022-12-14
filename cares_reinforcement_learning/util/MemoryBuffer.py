from collections import deque
import random


class MemoryBuffer:

    def __init__(self, max_capacity):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly sample experiences from buffer of size batch_size
        experience_batch = random.sample(self.buffer, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        states, actions, rewards, next_states, dones = zip(*experience_batch)

        return states, actions, rewards, next_states, dones


class MemoryBufferV2:

    def __init__(self, max_capacity):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        # Randomly sample experiences from buffer of size batch_size
        return random.sample(self.buffer, batch_size)
