"""
RepetitionManager class for handling episode repetition logic in training.
"""

import numpy as np


class EpisodeReplay:
    """
    A class that records actions during episodes and stores the best performing episode
    for replay. Updates the stored episode only when a new episode achieves a higher reward.
    """

    def __init__(self) -> None:
        self.current_actions: list[np.ndarray] = []
        self.best_actions: list[np.ndarray] = []

        self.best_reward: float = float("-inf")

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

        repeat = False
        # Check if this episode is better than the previous best

        if total_reward > self.best_reward:
            self.best_reward = total_reward
            self.best_actions = self.current_actions.copy()
            repeat = True

        self.current_actions = []

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


class RepetitionManager:
    """
    Manages episode repetition logic and state.

    Encapsulates the complex state machine for episode repetition to make
    the training loop cleaner and more maintainable.
    """

    def __init__(self, max_repetitions: int):
        """
        Initialize the RepetitionManager.

        Args:
            max_repetitions: Maximum number of times to repeat an episode
        """
        self.max_repetitions = max_repetitions
        self.enabled = max_repetitions > 0

        # Internal state
        self.buffer = EpisodeReplay()
        self.repeat = False
        self.is_repeating = False
        self.current_repetition_count = 0
        self.total_repetitions = 0  # For logging

    def record_action(self, action) -> None:
        """Record an action for potential repetition."""
        self.buffer.record_action(action)

    def should_repeat(self, episode_timesteps: int) -> bool:
        """
        Check if we should use repetition action instead of policy action.

        Args:
            episode_timesteps: Current timestep in episode

        Returns:
            True if we should use repetition action
        """
        return (
            self.enabled
            and self.repeat
            and self.buffer.has_best_episode()
            and episode_timesteps <= len(self.buffer.best_actions)
        )

    def get_repetition_action(self, episode_timesteps: int):
        """
        Get the action from the best episode replay.

        Args:
            episode_timesteps: Current timestep in episode (1-indexed)

        Returns:
            Action from the best episode
        """
        if not self.should_repeat(episode_timesteps):
            raise ValueError("Cannot get repetition action when not repeating")

        # EpisodeReplay expects 0-indexed
        action = self.buffer.replay_best_episode(episode_timesteps - 1)

        # Check if this is the last action in the sequence
        if episode_timesteps >= len(self.buffer.best_actions):
            self._stop_current_repetition()

        return action

    def finish_episode(self, episode_reward: float, in_training_phase: bool) -> None:
        """
        Handle episode completion and decide on repetition.

        Args:
            episode_reward: Reward achieved in this episode
            in_training_phase: Whether we're past exploration phase
        """
        if not self.enabled:
            return

        if self.is_repeating:
            self._handle_repetition_cycle()
        elif in_training_phase:
            # Check if we should start
            should_start_repeating = self.buffer.finish_episode(episode_reward)
            if should_start_repeating:
                self._start_repetition_cycle()
        else:
            self.buffer.finish_episode(episode_reward)

    def _start_repetition_cycle(self) -> None:
        """Start a new repetition cycle."""
        self.repeat = True
        self.is_repeating = True
        self.current_repetition_count = 0
        self.total_repetitions += 1

    def _handle_repetition_cycle(self) -> None:
        """Handle ongoing repetition cycle."""
        self.repeat = True
        self.current_repetition_count += 1
        if self.current_repetition_count >= self.max_repetitions:
            self._stop_repetition_cycle()

    def _stop_current_repetition(self) -> None:
        """Stop the current repetition (reached end of action sequence)."""
        self.repeat = False

    def _stop_repetition_cycle(self) -> None:
        """Stop the entire repetition cycle (reached max repetitions)."""
        self.repeat = False
        self.is_repeating = False
        self.current_repetition_count = 0

    def get_status_info(self) -> dict:
        """
        Get current repetition status for logging.

        Returns:
            Dictionary with repetition status information
        """
        return {
            "repeated": self.total_repetitions,
            "is_repeating": self.is_repeating,
            "repetition_count": self.current_repetition_count,
        }
