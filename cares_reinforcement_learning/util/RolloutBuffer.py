import logging
from collections import deque
import random

class RolloutBuffer:
    def __init__(self):
        self.states       = []
        self.actions      = []
        self.log_probs    = []
        self.next_states  = []
        self.rewards      = []
        self.dones        = []

    def clear(self):
        del self.states[:]
        del self.actions[:]
        del self.log_probs[:]
        del self.next_states[:]
        del self.rewards[:]
        del self.dones[:]