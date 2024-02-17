import logging
import random
from collections import deque
import numpy as np


class MemoryBuffer:
    def __init__(self, max_capacity=int(1e6)):
        self.buffer = deque([], maxlen=max_capacity)

    def add(self, *experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        batch_size = min(batch_size, len(self.buffer))
        experience_batch = random.sample(self.buffer, batch_size)
        # states, actions, rewards, next_states, dones = zip(*experience_batch)
        # return states, actions, rewards, next_states, dones
        transposed_batch = zip(*experience_batch)
        # return [component for component in transposed_batch]
        return transposed_batch

    def flush(self):
        states, actions, rewards, next_states, dones, log_probs = zip(
            *[(element[i] for i in range(len(element))) for element in self.buffer]
        )
        self.buffer.clear()
        return states, actions, rewards, next_states, dones, log_probs


# #----------------------------------------------------------------------------------------
# #----------------------------------------------------------------------------------------

# class MemoryBuffer:
#     def __init__(self, max_capacity=int(1e6)):
#         self.max_size = max_capacity
#         self.ptr = 0
#         self.size = 0
#         self.state = None
#
#     def add(self, state, action, reward, next_state, done):
#
#         if self.state is None:
#             # Infer dimensions from the inputs to avoid passing them as arg
#             state_dim =  state.shape[0]
#             action_dim = action.shape[0]

#             # Initialize arrays
#             self.state = np.empty((self.max_size, *state_dim))
#             self.action = np.zeros((self.max_size, action_dim))
#             self.next_state = np.zeros((self.max_size, state_dim))
#             self.reward = np.zeros((self.max_size))
#             self.done = np.zeros((self.max_size))
#
#         # Add the experience to the buffer
#         self.state[self.ptr] = state
#         self.action[self.ptr] = action
#         self.next_state[self.ptr] = next_state
#         self.reward[self.ptr] = reward
#         self.done[self.ptr] = done
#
#         self.ptr = (self.ptr + 1) % self.max_size
#         self.size = min(self.size + 1, self.max_size)
#
#     def sample(self, batch_size):
#         #batch_size = min(batch_size, self.size) revisa luego
#         ind = np.random.randint(0, self.size, size=batch_size)
#         return (
#             self.state[ind],
#             self.action[ind],
#             self.reward[ind],
#             self.next_state[ind],
#             self.done[ind],
#         )
