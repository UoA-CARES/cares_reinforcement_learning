import tempfile
from pathlib import Path

import numpy as np
import pytest

from cares_reinforcement_learning.memory.memory_buffer import (
    MARLMemoryBuffer,
    SARLMemoryBuffer,
)
from cares_reinforcement_learning.types.experience import (
    MultiAgentExperience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    SARLObservation,
)


def get_sarl_observation(state_size: int = 4) -> SARLObservation:
    """Create a dummy SARL observation for testing."""
    return SARLObservation(vector_state=np.array([1.0] * state_size), image_state=None)


def get_indexed_sarl_observation(index: int, state_size: int = 4) -> SARLObservation:
    """Create an indexed SARL observation for testing (uses index as state value)."""
    return SARLObservation(
        vector_state=np.array([float(index)] * state_size), image_state=None
    )


def get_marl_observation(state_size: int = 4, num_agents: int = 2) -> MARLObservation:
    """Create a dummy MARL observation for testing."""
    agent_states = {
        f"agent_{i}": np.array([1.0] * state_size) for i in range(num_agents)
    }
    return MARLObservation(
        global_state=np.array([1.0] * state_size),
        agent_states=agent_states,
        avail_actions=np.ones((num_agents,), dtype=bool),
    )


def get_indexed_marl_observation(
    index: int, state_size: int = 4, num_agents: int = 2
) -> MARLObservation:
    """Create an indexed MARL observation for testing (uses index as state value)."""
    agent_states = {
        f"agent_{i}": np.array([float(index)] * state_size) for i in range(num_agents)
    }
    return MARLObservation(
        global_state=np.array([float(index)] * state_size),
        agent_states=agent_states,
        avail_actions=np.ones((num_agents,), dtype=bool),
    )


def _images_the_same(image_one, image_two):
    if image_one is None and image_two is None:
        return True

    return image_one.shape == image_two.shape and not (
        np.bitwise_xor(image_one, image_two).any()
    )


def test_save_load_vector_sarl():
    """Test that SARL buffer save and load works correctly."""
    buffer = SARLMemoryBuffer(max_capacity=1000)
    data_size = 100

    for i in range(data_size):
        obs = get_indexed_sarl_observation(i)
        next_obs = get_indexed_sarl_observation(i)
        experience = SingleAgentExperience(
            observation=obs,
            next_observation=next_obs,
            action=np.array([float(i)]),
            reward=float(i),
            done=bool(i % 2),
            truncated=False,
            info={},
        )
        buffer.add(experience)

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer.save(temp_dir, "memory_buffer")
        loaded_buffer = SARLMemoryBuffer.load(temp_dir, "memory_buffer")

        # Check basic properties
        assert len(buffer) == len(loaded_buffer)
        assert buffer.max_capacity == loaded_buffer.max_capacity
        assert buffer.current_size == loaded_buffer.current_size
        assert buffer.tree_pointer == loaded_buffer.tree_pointer

        # Check beta values
        assert buffer.init_beta == loaded_buffer.init_beta
        assert buffer.beta == loaded_buffer.beta
        assert buffer.d_beta == loaded_buffer.d_beta

        # Check priority values
        assert buffer.min_priority == loaded_buffer.min_priority
        assert buffer.max_priority == loaded_buffer.max_priority

        # Check sum_tree levels
        sum_tree_levels = buffer.sum_tree.levels
        loaded_sum_tree_levels = loaded_buffer.sum_tree.levels
        assert len(sum_tree_levels) == len(loaded_sum_tree_levels)
        for i, _ in enumerate(sum_tree_levels):
            assert np.array_equal(sum_tree_levels[i], loaded_sum_tree_levels[i])

        # Check inverse_tree levels
        inverse_tree_levels = buffer.inverse_tree.levels
        loaded_inverse_tree_levels = loaded_buffer.inverse_tree.levels
        assert len(inverse_tree_levels) == len(loaded_inverse_tree_levels)
        for i, _ in enumerate(inverse_tree_levels):
            assert np.array_equal(inverse_tree_levels[i], loaded_inverse_tree_levels[i])

        # Verify sampled experiences match
        for experience, loaded_experience in zip(
            buffer.memory_buffers, loaded_buffer.memory_buffers
        ):
            assert np.allclose(
                experience.observation.vector_state,
                loaded_experience.observation.vector_state,
            )
            assert _images_the_same(
                experience.observation.image_state,
                loaded_experience.observation.image_state,
            )
            assert np.allclose(
                experience.next_observation.vector_state,
                loaded_experience.next_observation.vector_state,
            )
            assert _images_the_same(
                experience.next_observation.image_state,
                loaded_experience.next_observation.image_state,
            )
            assert np.allclose(experience.action, loaded_experience.action)
            assert np.isclose(experience.reward, loaded_experience.reward)
            assert experience.done == loaded_experience.done
            assert experience.truncated == loaded_experience.truncated


def test_save_load_vector_marl():
    """Test that MARL buffer save and load works correctly."""
    buffer = MARLMemoryBuffer(max_capacity=1000)
    data_size = 100

    for i in range(data_size):
        obs = get_indexed_marl_observation(i)
        next_obs = get_indexed_marl_observation(i)
        experience = MultiAgentExperience(
            observation=obs,
            next_observation=next_obs,
            action=[np.array([float(i)]), np.array([float(i)])],
            reward=[float(i), float(i)],
            done=[bool(i % 2), bool(i % 2)],
            truncated=[False, False],
            info={},
        )
        buffer.add(experience)

    with tempfile.TemporaryDirectory() as temp_dir:
        buffer.save(temp_dir, "memory_buffer")
        loaded_buffer = MARLMemoryBuffer.load(temp_dir, "memory_buffer")

        # Check basic properties
        assert len(buffer) == len(loaded_buffer)
        assert buffer.max_capacity == loaded_buffer.max_capacity
        assert buffer.current_size == loaded_buffer.current_size
        assert buffer.tree_pointer == loaded_buffer.tree_pointer

        # Check beta values
        assert buffer.init_beta == loaded_buffer.init_beta
        assert buffer.beta == loaded_buffer.beta
        assert buffer.d_beta == loaded_buffer.d_beta

        # Check priority values
        assert buffer.min_priority == loaded_buffer.min_priority
        assert buffer.max_priority == loaded_buffer.max_priority

        # Check sum_tree levels
        sum_tree_levels = buffer.sum_tree.levels
        loaded_sum_tree_levels = loaded_buffer.sum_tree.levels
        assert len(sum_tree_levels) == len(loaded_sum_tree_levels)
        for i, _ in enumerate(sum_tree_levels):
            assert np.array_equal(sum_tree_levels[i], loaded_sum_tree_levels[i])

        # Check inverse_tree levels
        inverse_tree_levels = buffer.inverse_tree.levels
        loaded_inverse_tree_levels = loaded_buffer.inverse_tree.levels
        assert len(inverse_tree_levels) == len(loaded_inverse_tree_levels)
        for i, _ in enumerate(inverse_tree_levels):
            assert np.array_equal(inverse_tree_levels[i], loaded_inverse_tree_levels[i])

        # Verify sampled experiences match
        # Verify sampled experiences match
        for experience, loaded_experience in zip(
            buffer.memory_buffers, loaded_buffer.memory_buffers
        ):
            assert np.allclose(
                experience.observation.global_state,
                loaded_experience.observation.global_state,
            )
            for agent_id in experience.observation.agent_states:
                assert np.allclose(
                    experience.observation.agent_states[agent_id],
                    loaded_experience.observation.agent_states[agent_id],
                )
            assert np.allclose(
                experience.next_observation.global_state,
                loaded_experience.next_observation.global_state,
            )
            for agent_id in experience.next_observation.agent_states:
                assert np.allclose(
                    experience.next_observation.agent_states[agent_id],
                    loaded_experience.next_observation.agent_states[agent_id],
                )
            for a1, a2 in zip(experience.action, loaded_experience.action):
                assert np.allclose(a1, a2)
            for r1, r2 in zip(experience.reward, loaded_experience.reward):
                assert np.isclose(r1, r2)
            for d1, d2 in zip(experience.done, loaded_experience.done):
                assert d1 == d2
            for t1, t2 in zip(experience.truncated, loaded_experience.truncated):
                assert t1 == t2
