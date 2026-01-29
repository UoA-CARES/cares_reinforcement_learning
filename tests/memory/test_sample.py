import pytest
import numpy as np

from cares_reinforcement_learning.types.experience import (
    SingleAgentExperience,
    MultiAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    SARLObservation,
    MARLObservation,
)
from cares_reinforcement_learning.memory.memory_buffer import (
    SARLMemoryBuffer,
    MARLMemoryBuffer,
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


def create_sarl_experience(
    observation: SARLObservation | None = None,
    next_observation: SARLObservation | None = None,
) -> SingleAgentExperience:
    """Create a SingleAgentExperience for testing."""
    if observation is None:
        observation = get_sarl_observation()
    if next_observation is None:
        next_observation = get_sarl_observation()

    return SingleAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=np.array([0.5]),
        reward=1.0,
        done=False,
        truncated=False,
        info={},
    )


def create_marl_experience(
    observation: MARLObservation | None = None,
    next_observation: MARLObservation | None = None,
) -> MultiAgentExperience:
    """Create a MultiAgentExperience for testing."""
    if observation is None:
        observation = get_marl_observation()
    if next_observation is None:
        next_observation = get_marl_observation()

    return MultiAgentExperience(
        observation=observation,
        next_observation=next_observation,
        action=[np.array([0.5]), np.array([0.3])],
        reward=[1.0, 0.5],
        done=[False, False],
        truncated=[False, False],
        info={},
    )


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_sample(buffer_class, exp_builder):
    """Test basic sampling methods (uniform, priority, inverse_priority)."""
    buffer = buffer_class(max_capacity=int(1e6))

    for i in range(4):
        experience = exp_builder()
        buffer.add(experience)

    batch_size = 3
    sample = buffer.sample_uniform(batch_size)

    assert len(sample.experiences) == len(sample.indices) == batch_size

    sample = buffer.sample_inverse_priority(batch_size)

    assert (
        len(sample.experiences)
        == len(sample.indices)
        == len(sample.weights)
        == batch_size
    )

    sample = buffer.sample_priority(batch_size)

    assert (
        len(sample.experiences)
        == len(sample.indices)
        == len(sample.weights)
        == batch_size
    )


@pytest.mark.parametrize(
    "buffer_class,exp_builder",
    [
        (SARLMemoryBuffer, create_sarl_experience),
        (MARLMemoryBuffer, create_marl_experience),
    ],
)
def test_sample_more_than_buffer(buffer_class, exp_builder):
    """Test sampling batch larger than buffer capacity."""
    buffer = buffer_class(max_capacity=5)

    for i in range(5):
        experience = exp_builder()
        buffer.add(experience)

    batch_size = 10
    sample = buffer.sample_uniform(batch_size)

    assert len(sample.experiences) == len(sample.indices) == len(sample.weights) == 5

    sample = buffer.sample_priority(batch_size)

    assert len(sample.experiences) == len(sample.indices) == len(sample.weights) == 5

    sample = buffer.sample_inverse_priority(batch_size)

    assert len(sample.experiences) == len(sample.indices) == len(sample.weights) == 5


def test_sample_values_sarl():
    """Test that SARL uniform sample values match what was added."""
    buffer = SARLMemoryBuffer(max_capacity=int(1e6))

    for i in range(10):
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

    sample = buffer.sample_uniform(1)

    index = 0
    value = sample.indices[index]
    # Verify the sampled observation matches the indexed state
    assert np.allclose(sample.experiences[index].observation.vector_state, float(value))
    assert np.allclose(
        sample.experiences[index].next_observation.vector_state, float(value)
    )
    assert np.allclose(sample.experiences[index].action, float(value))
    assert np.isclose(sample.experiences[index].reward, float(value))
    assert sample.experiences[index].done == bool(value % 2)
    assert sample.experiences[index].truncated is False
    assert sample.indices[index] == value


def test_sample_values_marl():
    """Test that MARL uniform sample values match what was added."""
    buffer = MARLMemoryBuffer(max_capacity=int(1e6))

    for i in range(10):
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

    sample = buffer.sample_uniform(1)

    index = 0
    value = sample.indices[index]
    # Verify the sampled observation matches the indexed state
    assert np.allclose(sample.experiences[index].observation.global_state, float(value))
    assert np.allclose(
        sample.experiences[index].next_observation.global_state, float(value)
    )
    assert sample.experiences[index].done[0] == bool(value % 2)
    assert sample.experiences[index].truncated[0] is False
    assert sample.indices[index] == value


def test_sample_priority_values_sarl():
    """Test that SARL priority sample values match what was added."""
    buffer = SARLMemoryBuffer(max_capacity=int(1e6))

    for i in range(10):
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

    sample = buffer.sample_priority(1)

    index = 0
    value = sample.indices[index]
    assert np.allclose(sample.experiences[index].observation.vector_state, float(value))
    assert np.allclose(
        sample.experiences[index].next_observation.vector_state, float(value)
    )
    assert np.allclose(sample.experiences[index].action, float(value))
    assert np.isclose(sample.experiences[index].reward, float(value))
    assert sample.experiences[index].done == bool(value % 2)
    assert sample.experiences[index].truncated is False
    assert sample.indices[index] == value
    assert sample.weights[index] == 1.0


def test_sample_priority_values_marl():
    """Test that MARL priority sample values match what was added."""
    buffer = MARLMemoryBuffer(max_capacity=int(1e6))

    for i in range(10):
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

    sample = buffer.sample_priority(1)

    index = 0
    value = sample.indices[index]
    assert np.allclose(sample.experiences[index].observation.global_state, float(value))
    assert np.allclose(
        sample.experiences[index].next_observation.global_state, float(value)
    )
    assert sample.experiences[index].done[0] == bool(value % 2)
    assert sample.experiences[index].truncated[0] is False
    assert sample.indices[index] == value
    assert sample.weights[index] == 1.0


def test_sample_inverse_priority_values_sarl():
    """Test that SARL inverse priority sample values and weights match."""
    buffer = SARLMemoryBuffer(max_capacity=int(1e6))
    size = 10

    for i in range(size):
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

    # Set all priorities to 1
    ind = np.arange(size)
    priorities = np.ones(size)
    buffer.update_priorities(ind, priorities)

    sample = buffer.sample_inverse_priority(1)

    index = 0
    value = sample.indices[index]
    assert np.allclose(sample.experiences[index].observation.vector_state, float(value))
    assert np.allclose(
        sample.experiences[index].next_observation.vector_state, float(value)
    )
    assert np.allclose(sample.experiences[index].action, float(value))
    assert np.isclose(sample.experiences[index].reward, float(value))
    assert sample.experiences[index].done == bool(value % 2)
    assert sample.experiences[index].truncated is False
    assert sample.indices[index] == value
    assert abs(sample.weights[index] - size) < 0.001


def test_sample_inverse_priority_values_marl():
    """Test that MARL inverse priority sample values and weights match."""
    buffer = MARLMemoryBuffer(max_capacity=int(1e6))
    size = 10

    for i in range(size):
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

    # Set all priorities to 1
    ind = np.arange(size)
    priorities = np.ones(size)
    buffer.update_priorities(ind, priorities)

    sample = buffer.sample_inverse_priority(1)

    index = 0
    value = sample.indices[index]
    assert np.allclose(sample.experiences[index].observation.global_state, float(value))
    assert np.allclose(
        sample.experiences[index].next_observation.global_state, float(value)
    )
    assert sample.experiences[index].done[0] == bool(value % 2)
    assert sample.experiences[index].truncated[0] is False
    assert sample.indices[index] == value
    assert abs(sample.weights[index] - size) < 0.001
