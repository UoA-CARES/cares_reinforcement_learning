"""
Example Implemtnations:
https://github.com/Howuhh/prioritized_experience_replay/blob/main/memory/buffer.py
https://github.com/sfujim/LAP-PAL/blob/master/continuous/utils.py

"""

import os
import pickle
import random
import tempfile
from abc import ABC, abstractmethod
from collections import deque
from dataclasses import dataclass
from typing import Deque, Generic, TypeVar

import numpy as np

from cares_reinforcement_learning.memory.sum_tree import SumTree
from cares_reinforcement_learning.types.experience import (
    ExperienceType,
    MultiAgentExperience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import Observation

ObsType = TypeVar("ObsType", bound=Observation)
ExpType = TypeVar("ExpType", bound=ExperienceType)
MemBufT = TypeVar("MemBufT", bound="MemoryBuffer")


@dataclass(frozen=True, slots=True)
class Sample(Generic[ExpType]):
    experiences: list[ExpType]
    indices: list[int]
    weights: np.ndarray


class MemoryBuffer(ABC, Generic[ExpType]):
    """
    A prioritized replay buffer implementation for reinforcement learning.

    This buffer stores experiences and allows for efficient sampling based on priorities.
    Experiences are stored in the order: state, action, reward, next_state, done, ...

    Args:
        max_capacity (int): The maximum capacity of the buffer. Default is 1e6.
        min_priority (float): The minimum priority value. Default is 1e-4 - just above 0.
        beta (float): The initial value of the beta parameter for importance weight calculation. Default is 0.4.
        d_beta (float): The rate of change for the beta parameter. Default is 6e-7 - presumned over 1,000,000 steps.
        n_step (int): The number of steps to use for n-step learning. Default is 1.
        gamma (float): The discount factor for n-step learning. Default is 0.99.

    Attributes:
        priority_params (dict): Additional parameters for priority calculation.
        max_capacity (int): The maximum capacity of the buffer.
        current_size (int): The current size of the buffer.
        memory_buffers (list): An array of buffers for each experience type.
        sum_tree (SumTree): The SumTree data structure for efficient sampling based on priorities.
        inverse_tree (SumTree): The SumTree data structure for efficient sampling based on inverse priorities.
        tree_pointer (int): The location to add the next item into the tree.
        min_priority (float): The minimum priority value.
        init_beta (float): The initial value of the beta parameter.
        beta (float): The current value of the beta parameter.
        d_beta (float): The rate of change for the beta parameter.
        max_priority (float): The maximum priority value in the buffer.

    Methods:
        __len__(): Returns the current size of the buffer.
        add(state, action, reward, next_state, done, *extra): Adds a single experience to the buffer.
        sample_uniform(batch_size): Samples experiences uniformly from the buffer.
        _importance_sampling_prioritised_weights(indices, weight_normalisation): Calculates the importance-sampling weights for prioritized replay.
        sample_priority(batch_size, sampling, weight_normalisation): Samples experiences from the buffer based on priorities.
        sample_inverse_priority(batch_size): Samples experiences from the buffer based on inverse priorities.
        update_priorities(indices, priorities): Updates the priorities of the buffer at the given indices.
        flush(): Flushes the memory buffers and returns the experiences in order.
        sample_consecutive(batch_size): Randomly samples consecutive experiences from the memory buffer.
        get_statistics(): Returns statistics about the buffer.
        clear(): Clears the buffer.
    """

    def __init__(
        self,
        max_capacity: int = int(1e6),
        min_priority: float = 1e-4,
        beta: float = 0.4,
        d_beta: float = 6e-7,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        # pylint: disable-next=unused-argument

        self.max_capacity = max_capacity

        # size is the current size of the buffer
        self.current_size = 0

        # Functionally is an array of buffers for each experience type
        self.memory_buffers: list[ExpType] = []  # type: ignore

        # n-step learning
        self.n_step = n_step
        self.n_step_buffer: Deque[ExpType] = deque(maxlen=self.n_step)
        self.gamma = gamma

        # The SumTree is an efficient data structure for sampling based on priorities
        self.sum_tree = SumTree(self.max_capacity)
        self.inverse_tree = SumTree(self.max_capacity)

        # The location to add the next item into the tree - index for the SumTree
        self.tree_pointer = 0

        # Minimum priroity (aka epsilon), prevents zero probabilities
        self.min_priority = min_priority

        # Determines the amount of importance-sampling correction, b = 1 fully compensate for the non-uniform probabilities
        self.init_beta = beta
        self.beta = self.init_beta
        self.d_beta = d_beta

        # Current max priority
        self.max_priority = 1.0

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.current_size

    @abstractmethod
    def _apply_n_step(self, n_step_buffer: Deque[ExpType]) -> ExpType: ...

    def add(self, experience: ExpType) -> None:
        """
        Adds a single experience to the prioritized replay buffer.

        Data is expected to be stored in the order: state, action, reward, next_state, done, ...

        Args:
            state: The current state of the environment.
            action: The action taken in the current state.
            reward: The reward received for taking the action.
            next_state: The next state of the environment after taking the action.
            done: A flag indicating whether the episode is done after taking the action.
            *extra: Extra is a variable list of extra experience data to be added (e.g. log_prob).

        Returns:
            None
        """
        # n-step learning - default is 1-step which means regular buffer behaviour
        self.n_step_buffer.append(experience.clone())  # type: ignore
        if len(self.n_step_buffer) < self.n_step:
            return

        # Calculate the n-step return - default is 1-step which means regular buffer behaviour
        experience = self._apply_n_step(self.n_step_buffer)

        if len(self.memory_buffers) < self.max_capacity:
            self.memory_buffers.append(experience)
        else:
            self.memory_buffers[self.tree_pointer] = experience

        # Add the priority to the SumTree - Prioritised Experience Replay
        new_priority = self.max_priority
        self.sum_tree.set(self.tree_pointer, new_priority)

        self.tree_pointer = (self.tree_pointer + 1) % self.max_capacity
        self.current_size = min(self.current_size + 1, self.max_capacity)

    def sample_uniform(self, batch_size: int) -> Sample[ExpType]:
        """
        Samples experiences uniformly from the buffer.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            Sample[ExpType]: A sample of experiences.
        """

        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)
        indices = np.random.randint(self.current_size, size=batch_size)

        # Extracts the experiences at the desired indices from the buffer
        experiences = [self.memory_buffers[i] for i in indices]

        return Sample(
            experiences=experiences,
            indices=indices.tolist(),
            weights=np.ones(len(experiences)),
        )

    def _importance_sampling_prioritised_weights(
        self, indices: np.ndarray, weight_normalisation="batch"
    ) -> np.ndarray:
        """
        Calculates the importance-sampling weights for prioritized replay and prioritises based on population max.

        PER Paper: https://arxiv.org/pdf/1511.05952.pdf

        Args:
            indices (np.ndarray): A list of indices representing the transitions to calculate weights for.
            weight_normalisation (str): The type of weight normalisation to use. Options are "batch" or "population".

        Returns:
            np.ndarray: An array of importance-sampling weights.

        Notes:
            - The importance-sampling weights are used to compensate for the non-uniform probabilities of sampling transitions.
            - The weights are calculated using the formula w_i = (1/N * 1/P(i))^β, where N is the current size of the replay buffer,
              P(i) is the priority of transition i, and β is a hyperparameter.
            - The weights are then normalized by dividing them by the maximum weight to ensure stability.
        """

        max_value = self.sum_tree.levels[0][0]

        priorities = self.sum_tree.levels[-1][indices]
        probabilities = priorities / max_value

        weights = (probabilities * self.current_size) ** (-self.beta)

        max_weight = 1.0
        # Batch normalisation is the default and normalises the weights by the maximum weight in the batch
        if weight_normalisation == "batch":
            max_weight = weights.max()
        # Population normalisation normalises the weights by the maximum weight in the population (buffer)
        elif weight_normalisation == "population":
            p_min = (
                self.sum_tree.levels[-1][: self.current_size].min()
                / self.sum_tree.levels[0][0]
            )
            max_weight = (p_min * self.current_size) ** (-self.beta)

        weights /= max_weight

        return weights

    def sample_priority(
        self,
        batch_size: int,
        sampling_strategy: str = "stratified",
        weight_normalisation: str = "batch",
    ) -> Sample[ExpType]:
        """
        Samples experiences from the prioritized replay buffer.

        Stratifed vs Simple: https://www.sagepub.com/sites/default/files/upm-binaries/40803_5.pdf

        Args:
            batch_size (int): The number of experiences to sample.
            sampling_stratagy (str): The sampling strategy to use. Options are "simple" or "stratified".
            weight_normalisation (str): The type of weight normalisation to use. Options are "batch" or "population".

        Returns:
            list[ExpType]: A list of sampled experiences based on inverse priorities.
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the importance weights for each sampled experience.
        """
        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        if sampling_strategy == "simple":
            indices = self.sum_tree.sample_simple(batch_size)
        elif sampling_strategy == "stratified":
            indices = self.sum_tree.sample_stratified(batch_size)
        else:
            raise ValueError(f"Unknown sampling scheme: {sampling_strategy}")

        weights = self._importance_sampling_prioritised_weights(
            indices, weight_normalisation=weight_normalisation
        )

        # We therefore exploit the flexibility of annealing the amount of importance-sampling
        # correction over time, by defining a schedule on the exponent β that reaches 1 only at the end of
        # learning. In practice, we linearly anneal β from its initial value β0 to 1. Note that the choice of this
        # hyperparameter interacts with choice of prioritization exponent α; increasing both simultaneously
        # prioritizes sampling more aggressively at the same time as correcting for it more strongly.
        self.beta = min(self.beta + self.d_beta, 1.0)

        # Extracts the experiences at the desired indices from the buffer
        experiences = [self.memory_buffers[i] for i in indices]

        return Sample(
            experiences=experiences,
            indices=indices.tolist(),
            weights=weights,
        )

    def sample_inverse_priority(self, batch_size: int) -> Sample[ExpType]:
        """
        Samples experiences from the buffer based on inverse priorities.

        Args:
            batch_size (int): The number of experiences to sample.

        Returns:
            list[ExpType]: A list of sampled experiences based on inverse priorities.
                - The indices represent the indices of the sampled experiences in the buffer.
                - The weights represent the inverse importance weights for each sampled experience.

        """
        if self.current_size == 0:
            return Sample(
                experiences=[],
                indices=[],
                weights=np.asarray([]),
            )

        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size)

        top_value = self.sum_tree.levels[0][0]

        # TODO add inverse (1 - prob into SumTree instead)
        # Inverse based on paper for LA3PD - https://arxiv.org/abs/2209.00532
        reversed_priorities = top_value / (
            self.sum_tree.levels[-1][: self.current_size] + 1e-6
        )

        self.inverse_tree.batch_set(np.arange(self.current_size), reversed_priorities)

        indices = self.inverse_tree.sample_simple(batch_size)

        # Extracts the experiences at the desired indices from the buffer
        experiences = [self.memory_buffers[i] for i in indices]

        return Sample(
            experiences=experiences,
            indices=indices.tolist(),
            weights=reversed_priorities[indices],
        )

    def reset_priorities(self) -> None:
        """
        Resets all priorities in the replay buffer to the maximum priority.

        This is useful in scenarios where the priorities need to be re-evaluated or when initializing the buffer.
        """
        self.sum_tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        for i in range(self.current_size):
            self.sum_tree.set(i, self.max_priority)

    def reset_max_priority(self) -> None:
        """
        Recalculates max_priority to the actual maximum priority currently in the buffer.

        This prevents priority inflation over time by setting max_priority to the current
        maximum value in the buffer rather than an accumulated historical maximum.

        This is useful when the priority distribution has changed significantly and
        the historical max_priority no longer reflects the current state of the buffer.
        """
        if self.current_size > 0:
            current_priorities = self.sum_tree.levels[-1][: self.current_size]
            self.max_priority = float(current_priorities.max())
        else:
            self.max_priority = 1.0

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """
        Update the priorities of the replay buffer at the given indices.

        Parameters:
        - indices (array-like): The indices of the replay buffer to update.
        - priorities (array-like): The new priorities to assign to the specified indices.

        Returns:
        None
        """
        self.max_priority = max(priorities.max(), self.max_priority)
        self.sum_tree.batch_set(indices, priorities)

    def flush(self) -> Sample[ExpType]:
        """
        Flushes the memory buffers and returns the experiences in order.

        Returns:
            experiences (list): The full memory buffer in order.
        """
        if self.current_size == 0:
            return Sample(
                experiences=[],
                indices=[],
                weights=np.asarray([]),
            )

        sample = Sample(
            experiences=self.memory_buffers[: self.current_size],
            indices=list(range(self.current_size)),
            weights=(np.asarray([1.0] * self.current_size)),
        )

        self.clear()

        return sample

    def sample_consecutive(
        self, batch_size: int
    ) -> tuple[Sample[ExpType], Sample[ExpType]]:
        """
        Randomly samples consecutive experiences from the memory buffer.

        Args:
            batch_size (int): The number of consecutive experiences to sample.

        Returns:
            tuple: A tuple containing the sampled experiences and their corresponding next experiences.
                - list[ExpType]: A list of sampled experiences.
                - list[ExpType]: A list of next sampled experiences.

        """
        if self.current_size == 0:
            empty: Sample[ExpType] = Sample(
                experiences=[],
                indices=[],
                weights=np.asarray([]),
            )
            return empty, empty

        # If batch size is greater than size we need to limit it to just the data that exists
        batch_size = min(batch_size, self.current_size - 1)

        candididate_indices = list(range(self.current_size - 1))

        # A list of candidate indices includes all indices.
        sampled_indices: list[int] = []  # randomly sampled indices that is okay.
        # In this way, the sampling time depends on the batch size rather than buffer size.

        # Add in only experiences that are not done and not already sampled.
        while len(sampled_indices) < batch_size:
            # Sample size based on how many still needed.
            idxs = random.sample(candididate_indices, batch_size - len(sampled_indices))
            for i in idxs:
                # Check the experience is not done and not already sampled.
                done = self.memory_buffers[i].done_flag
                if (not done) and (i not in sampled_indices):
                    sampled_indices.append(i)

        experiences = [self.memory_buffers[i] for i in sampled_indices]

        sample = Sample(
            experiences=experiences,
            indices=sampled_indices,
            weights=np.asarray([1.0] * batch_size),
        )

        next_sampled_indices = (np.array(sampled_indices) + 1).tolist()

        experiences = [self.memory_buffers[i] for i in next_sampled_indices]

        next_sample = Sample(
            experiences=experiences,
            indices=next_sampled_indices,
            weights=np.asarray([1.0] * batch_size),
        )

        return sample, next_sample

    def get_statistics(self) -> dict[str, np.ndarray]:
        """
        Calculate statistics of the replay buffer.

        Returns:
            statistics (dict): A dictionary containing the following statistics:
                - observation_mean: Mean of the observations in the replay buffer.
                - observation_std: Standard deviation of the observations in the replay buffer.
                - delta_mean: Mean of the differences between consecutive observations.
                - delta_std: Standard deviation of the differences between consecutive observations.
        """
        states = np.array(self.memory_buffers[0][: self.current_size].tolist())
        next_states = np.array(self.memory_buffers[3][: self.current_size].tolist())
        diff_states = next_states - states

        # Add a small number to avoid zeros.
        observation_mean = np.mean(states, axis=0) + 0.00001
        observation_std = np.std(states, axis=0) + 0.00001
        delta_mean = np.mean(diff_states, axis=0) + 0.00001
        delta_std = np.std(diff_states, axis=0) + 0.00001

        statistics = {
            "observation_mean": observation_mean,
            "observation_std": observation_std,
            "delta_mean": delta_mean,
            "delta_std": delta_std,
        }
        return statistics

    def clear(self) -> None:
        """
        Clears the prioritised replay buffer.

        Resets the pointer, size, memory buffers, sum tree, max priority, and beta values.
        """
        self.tree_pointer = 0
        self.current_size = 0
        self.memory_buffers = []

        self.sum_tree = SumTree(self.max_capacity)
        self.max_priority = 1.0
        self.beta = self.init_beta

    def save(self, filepath: str, file_name: str) -> None:
        final_path = os.path.join(filepath, f"{file_name}.pkl")

        # create temp file in the same directory (so os.replace is atomic)
        with tempfile.NamedTemporaryFile("wb", dir=filepath, delete=False) as tmp:
            try:
                pickle.dump(self, tmp)
                tmp.flush()
                os.fsync(tmp.fileno())  # ensure data is written to disk
            except Exception:
                os.remove(tmp.name)  # cleanup on failure
                raise

        # atomically replace old file with new one
        os.replace(tmp.name, final_path)

    @classmethod
    def load(cls: type[MemBufT], file_path: str, file_name: str) -> MemBufT:
        """
        Simple object deserialization given a filename
        """
        with open(f"{file_path}/{file_name}.pkl", "rb") as f:
            obj = pickle.load(f)
            return obj


class SARLMemoryBuffer(MemoryBuffer[SingleAgentExperience]):
    def __init__(
        self,
        max_capacity: int = int(1e6),
        min_priority: float = 1e-4,
        beta: float = 0.4,
        d_beta: float = 6e-7,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(max_capacity, min_priority, beta, d_beta, n_step, gamma)

    def _apply_n_step(
        self, n_step_buffer: deque[SingleAgentExperience]
    ) -> SingleAgentExperience:
        first = n_step_buffer[0]
        last = n_step_buffer[-1]

        n_reward = last.reward
        n_next_observation = last.next_observation
        n_done = last.done
        n_truncated = last.truncated

        # Walk backwards over earlier transitions (excluding last)
        for n_exp in reversed(list(n_step_buffer)[:-1]):
            step_terminal = n_exp.done or n_exp.truncated

            # If a terminal happened at this step, the effective next_state becomes this step's next_state
            # and the terminal flags become this step's flags.
            if step_terminal:
                n_next_observation = n_exp.next_observation
                n_done = n_exp.done
                n_truncated = n_exp.truncated

            # Discounted accumulation. If terminal has occurred at/after this step,
            # we should not bootstrap further (i.e., multiply future return by 0).
            n_reward = n_exp.reward + self.gamma * n_reward * (1 - int(step_terminal))

        # Build new experience using state/action from the first transition
        return SingleAgentExperience(
            observation=first.observation,
            next_observation=n_next_observation,
            action=first.action,
            reward=n_reward,
            done=n_done,
            truncated=n_truncated,
            info=first.info,
        )


class MARLMemoryBuffer(MemoryBuffer[MultiAgentExperience]):
    def __init__(
        self,
        max_capacity: int = int(1e6),
        min_priority: float = 1e-4,
        beta: float = 0.4,
        d_beta: float = 6e-7,
        n_step: int = 1,
        gamma: float = 0.99,
    ):
        super().__init__(max_capacity, min_priority, beta, d_beta, n_step, gamma)

    def _apply_n_step(
        self,
        n_step_buffer: deque[MultiAgentExperience],
    ) -> MultiAgentExperience:
        first = n_step_buffer[0]
        last = n_step_buffer[-1]

        # Start from the last step in the window
        n_reward = list(last.reward)
        n_next_observation = last.next_observation
        n_done = list(last.done)
        n_truncated = list(last.truncated)

        # Walk backwards over earlier transitions (excluding last)
        for n_exp in reversed(list(n_step_buffer)[:-1]):
            # WARNING: This currently assumes that all agents share the same done/truncated flags
            step_terminal = n_exp.done_flag or n_exp.truncated_flag

            # discounted accumulation per agent
            if step_terminal:
                n_next_observation = n_exp.next_observation
                n_done = list(n_exp.done)
                n_truncated = list(n_exp.truncated)

            n_reward = [
                r + self.gamma * nr * (1 - int(step_terminal))
                for r, nr in zip(n_exp.reward, n_reward)
            ]

        return MultiAgentExperience(
            observation=first.observation,
            next_observation=n_next_observation,
            action=first.action,
            reward=n_reward,
            done=n_done,
            truncated=n_truncated,
            info=first.info,
        )


Memory = SARLMemoryBuffer | MARLMemoryBuffer
