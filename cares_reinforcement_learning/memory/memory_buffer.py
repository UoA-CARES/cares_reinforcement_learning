import logging
import random
from collections import deque

class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, state, action, reward, next_state, done):
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)


    def sample(self, batch_size):
        experience_batch = random.sample(self.buffer, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        states, actions, rewards, next_states, dones = zip(*experience_batch)

        return states, actions, rewards, next_states, dones