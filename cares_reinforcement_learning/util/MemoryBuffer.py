from collections import deque
import random

class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):
        self.max_capacity = max_capacity
        self.buffer = deque([], maxlen=self.max_capacity)

    def add(self, **experience):
        state      = experience["state"]
        action     = experience["action"]
        reward     = experience["reward"]
        next_state = experience["next_state"]
        done       = experience["done"]
        self.buffer.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        # Randomly sample experiences from buffer of size batch_size
        experience_batch = random.sample(self.buffer, batch_size)

        # Destructure batch experiences into tuples of _
        # eg. tuples of states, tuples of actions...
        states, actions, rewards, next_states, dones = zip(*experience_batch)

        return states, actions, rewards, next_states, dones
    
    def clear(self):
        self.buffer = deque([], maxlen=self.max_capacity)