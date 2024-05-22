import random

import numpy as np

from cares_reinforcement_learning.memory import SumTree


class EpisodicBuffer:

    def __init__(self, max_capacity: int = int(1e6)):
        self.max_capacity = max_capacity

        # size is the current size of the buffer
        self.current_size = 0

        # Functionally is an array of buffers for each experience type
        self.memory_buffers = []

        # episodic parameters
        self.full = False

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The current size of the buffer.
        """
        return self.current_size

    def add(self, experience) -> None:
        self.memory_buffers.append(experience)
        self.current_size += 1

    def add_episode(self, episodic_batch) -> None:
        """
        Add an episode to the episodic memory buffer.
        """
        if not self.full:
            # Add the episode to the buffer
            for i in range(len(episodic_batch[2])):
                self.add([episodic_batch[0], episodic_batch[1], episodic_batch[2][i], episodic_batch[3][i], episodic_batch[4][i],
                     episodic_batch[5][i], episodic_batch[6][i], episodic_batch[7][i], episodic_batch[8][i]])
                self.current_size += 1
            if self.current_size >= self.max_capacity:
                self.full = True

    def replaceEpisode(self, deleted_episode_id, deleted_episode_reward, episodic_batch):
        """
        Replace an episode in the episodic memory with a new episode.
        """
        self.delete_episode(deleted_episode_id, deleted_episode_reward)
        # Replace the episode with the new episode
        self.add_episode(episodic_batch)

    def delete_episode(self, episode_id, episode_reward):
        """
        Delete an episode from the episodic memory.
        """
        # Filter and keep episodes that do not match the target episode_id
        self.memory_buffers = [
            episode for episode in self.memory_buffers if episode[0] != episode_id and episode[1] != episode_reward]

    def sample_episode(self, target_episode_id, target_episode_reward, batch_size):
       
        matching_experiences = [
            experience for experience in self.memory_buffers if experience[0] == target_episode_id and experience[1] == target_episode_reward]

        if len(matching_experiences) >= batch_size:
            sampled_experiences = random.sample(
                matching_experiences, batch_size)
            """
            # Sort experiences by episode_step in descending order
            sorted_experiences = sorted(
                matching_experiences, key=lambda x: x[], reverse=True)

            # Select the top batch_size experiences with the highest episode_step
            sampled_experiences = sorted_experiences[:batch_size] 
            """
        else:
            sampled_experiences = matching_experiences

        # Unzip the experiences into separate lists
        _,_, states, actions, rewards, next_states, dones, episode_nums, episode_steps = zip(
            *sampled_experiences)
        return states, actions, rewards, next_states, dones, episode_nums, episode_steps

    def is_full(self):
        """
        Check if the episodic memory is full.
        """
        return self.current_size >= self.max_capacity


