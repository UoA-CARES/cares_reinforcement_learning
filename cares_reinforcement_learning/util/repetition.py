from typing import List

import numpy as np


class EpisodeReplay:
    """
    A class that records actions during episodes and stores the best performing episode
    for replay. Updates the stored episode only when a new episode achieves a higher reward.
    """

    def __init__(self) -> None:
        self.current_actions: List[np.ndarray] = []
        self.best_actions: List[np.ndarray] = []

        self.best_reward: float = float("-inf")
        self.current_reward: float = 0.0

    def start_episode(self) -> None:
        self.current_actions = []
        self.current_reward = 0.0

    def record_action(self, action: np.ndarray) -> None:
        """
        Record an action taken during the current episode.

        Args:
            action: The action taken (numpy array)
        """
        self.current_actions.append(action)

    def finish_episode(self, total_reward: float) -> bool:
        """
        Finish the current episode and update best episode if reward is higher.

        Args:
            total_reward: The total reward achieved in the current episode

        Returns:
            bool: True if this episode became the new best episode, False otherwise
        """
        self.current_reward = total_reward

        repeat = False
        # Check if this episode is better than the previous best

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_actions = self.current_actions.copy()
            repeat = True

        self.start_episode()  # Prepare for the next episode

        return repeat

    def has_best_episode(self) -> bool:
        """
        Check if we have recorded at least one complete episode.

        Returns:
            bool: True if we have a best episode stored, False otherwise
        """
        return len(self.best_actions) > 0

    def replay_best_episode(self, step: int) -> np.ndarray | None:
        """
        Returns the action taken at the specified step of the best episode, if available.
        Args:
            step (int): The index of the step within the best episode to retrieve the action for.
        Returns:
            np.ndarray | None: The action taken at the given step as a NumPy array if the best episode exists and the step is valid;
            otherwise, returns None.
        """

        if self.has_best_episode() and 0 <= step < len(self.best_actions):
            return self.best_actions[step]
        return None

    def reset(self) -> None:
        """Reset the replay system to initial state."""
        self.current_actions = []
        self.best_actions = []
        self.best_reward = float("-inf")
        self.current_reward = 0.0
