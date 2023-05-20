import MemoryBuffer


class RolloutBuffer(MemoryBuffer):
    """
    This class represents a rollout buffer used in Reinforcement Learning (RL).
    It inherits from the MemoryBuffer base class and overrides necessary methods
    to add and manage experiences with log_probs.

    The buffer stores experiences in lists and flushes all experiences when it's needed.
    """

    def __init__(self, max_capacity=int(1e6)):
        """
        The constructor for RolloutBuffer class.

        Parameters
        ----------
        max_capacity : int
            The maximum capacity of the buffer (default is 1,000,000).
        """
        super().__init__(max_capacity)
        self.buffers["log_probs"] = []

    def flush(self):
        """
        Flushes all the buffers and returns all experiences.

        Returns
        -------
        dict
            A dictionary of all experiences. Keys are the names of the buffers,
            and values are the lists of experiences.
        """
        experiences = {key: list(buffer) for key, buffer in self.buffers.items()}
        self.clear()
        return experiences
