


class RolloutBuffer:
    def __init__(self):
      self.states       = []
      self.actions      = []
      self.next_states  = []
      self.rewards      = []
      self.dones        = []
      self.log_probs    = []

    def add(self, **experience):
      self.states.append(experience["state"])
      self.actions.append(experience["action"])
      self.rewards.append(experience["reward"])
      self.next_states.append(experience["next_state"])
      self.dones.append(experience["done"])
      self.log_probs.append(experience["log_prob"])

    def clear(self):
      self.states       = []
      self.actions      = []
      self.next_states  = []
      self.rewards      = []
      self.dones        = []
      self.log_probs    = []
