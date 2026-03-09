class LinearScheduler:
    def __init__(self, start_value: float, end_value: float, decay_steps: int):
        self.start_value = start_value
        self.end_value = end_value
        self.decay_steps = decay_steps
        self.value = start_value

    def get_value(self, step: int) -> float:
        if step < self.decay_steps:
            self.value = self.start_value - (self.start_value - self.end_value) * (
                step / self.decay_steps
            )
        else:
            self.value = self.end_value
        return self.value


class ExponentialScheduler:
    def __init__(self, start_value: float, end_value: float, decay_steps: int):
        self.start = start_value
        self.end = end_value
        self.decay_steps = decay_steps
        if decay_steps == 0:
            self.gamma = 0.0
        else:
            self.gamma = (end_value / start_value) ** (1.0 / decay_steps)

    def get_value(self, step: int) -> float:
        return max(self.end, self.start * (self.gamma**step))
