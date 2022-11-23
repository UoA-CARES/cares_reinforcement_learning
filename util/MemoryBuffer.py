from collections import deque
import random
import torch
import numpy as np


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

        # Convert from _ tuples to _ tensors
        # e.g. states tuple to states tensor
        states = torch.FloatTensor(np.asarray(states))
        actions = torch.FloatTensor(np.asarray(actions))
        rewards = torch.FloatTensor(np.asarray(rewards))
        next_states = torch.FloatTensor(np.asarray(next_states))
        dones = torch.FloatTensor(np.asarray(dones))

        return states, actions, rewards, next_states, dones
