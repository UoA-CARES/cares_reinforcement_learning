import numpy as np
import pytest
import torch

from cares_reinforcement_learning.memory import memory_sampler
from cares_reinforcement_learning.memory.memory_buffer import (
    MARLMemoryBuffer,
    Sample,
    SARLMemoryBuffer,
)
from cares_reinforcement_learning.types.experience import (
    MultiAgentExperience,
    SingleAgentExperience,
)
from tests.memory.helpers import (
    create_marl_experience,
    create_sarl_experience,
    get_indexed_marl_observation,
    get_indexed_sarl_observation,
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


def _make_sequence_sarl_experience(i, done=False, truncated=False):
    return SingleAgentExperience(
        observation=get_indexed_sarl_observation(i),
        next_observation=get_indexed_sarl_observation(i + 1),
        action=np.array([float(i)], dtype=np.float32),
        reward=float(i + 1),
        done=done,
        truncated=truncated,
        info={},
        train_data={"step": i},
    )


def _make_sequence_marl_experience(i, done=False, truncated=False):
    observation = get_indexed_marl_observation(i)
    agents = list(observation.agent_states.keys())

    return MultiAgentExperience(
        observation=observation,
        next_observation=get_indexed_marl_observation(i + 1),
        action={agent: np.array([float(i)], dtype=np.float32) for agent in agents},
        reward={agent: float(i + 1) for agent in agents},
        done={agent: done for agent in agents},
        truncated={agent: truncated for agent in agents},
        info={},
        train_data={"step": i},
    )


def _force_sequence_start(monkeypatch, buffer, index, prioritized=False):
    """Force a particular starting index for deterministic sequence tests."""

    sample = Sample(
        experiences=[buffer.memory_buffers[index]],
        indices=[index],
        weights=np.ones(1),
    )

    method = "sample_priority" if prioritized else "sample_uniform"
    monkeypatch.setattr(buffer, method, lambda *args, **kwargs: sample)


@pytest.mark.parametrize(
    "buffer_class,builder",
    [
        (SARLMemoryBuffer, _make_sequence_sarl_experience),
        (MARLMemoryBuffer, _make_sequence_marl_experience),
    ],
)
def test_sample_sequences_order(monkeypatch, buffer_class, builder):
    """Sequences contain the correct consecutive experiences."""

    buffer = buffer_class(max_capacity=20)

    for i in range(10):
        buffer.add(builder(i))

    _force_sequence_start(monkeypatch, buffer, 2)

    sample = buffer.sample_sequences(batch_size=1, horizon=5)

    assert sample.indices == [2]
    assert len(sample.sequences) == 1

    steps = [exp.train_data["step"] for exp in sample.sequences[0]]

    assert steps == [2, 3, 4, 5, 6]
    np.testing.assert_array_equal(sample.weights, [1.0])


@pytest.mark.parametrize(
    "buffer_class,builder",
    [
        (SARLMemoryBuffer, _make_sequence_sarl_experience),
        (MARLMemoryBuffer, _make_sequence_marl_experience),
    ],
)
@pytest.mark.parametrize("boundary", ["done", "truncated"])
def test_sample_sequences_episode_boundary(
    monkeypatch, buffer_class, builder, boundary
):
    """Sequences include the terminal transition but never cross episodes."""

    buffer = buffer_class(max_capacity=20)

    for i in range(10):
        buffer.add(builder(i, **{boundary: i == 3}))

    _force_sequence_start(monkeypatch, buffer, 1)

    sample = buffer.sample_sequences(batch_size=1, horizon=5)

    steps = [exp.train_data["step"] for exp in sample.sequences[0]]

    assert steps == [1, 2, 3]

    # Starting directly on the terminal transition is also valid.
    _force_sequence_start(monkeypatch, buffer, 3)

    sample = buffer.sample_sequences(batch_size=1, horizon=5)

    assert len(sample.sequences[0]) == 1
    assert sample.sequences[0][0].train_data["step"] == 3


@pytest.mark.parametrize(
    "boundary,expected_terminated,expected_truncated",
    [
        ("done", 1.0, 0.0),
        ("truncated", 0.0, 1.0),
    ],
)
def test_sample_sequences_sarl_tensors(
    monkeypatch, boundary, expected_terminated, expected_truncated
):
    """Check SARL tensor shapes, padding, boundaries and train_data."""

    buffer = SARLMemoryBuffer(max_capacity=20)

    for i in range(6):
        buffer.add(
            _make_sequence_sarl_experience(
                i,
                **{boundary: i == 2},
            )
        )

    _force_sequence_start(monkeypatch, buffer, 1)

    batch, indices = memory_sampler.sample_sequences(
        memory=buffer,
        batch_size=1,
        horizon=5,
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(indices, [1])

    assert batch.observations.vector_state.shape[1] == 6
    assert batch.actions.shape == (1, 5, 1)
    assert batch.rewards.shape == (1, 5, 1)
    assert batch.valid.shape == (1, 5, 1)
    assert batch.weights.shape == (1, 1)

    # Only transitions 1 and 2 are real.
    np.testing.assert_array_equal(
        batch.valid[0, :, 0].numpy(),
        [1, 1, 0, 0, 0],
    )

    np.testing.assert_array_equal(
        batch.rewards[0, :, 0].numpy(),
        [2, 3, 0, 0, 0],
    )

    assert batch.terminated[0, 1, 0].item() == expected_terminated
    assert batch.truncated[0, 1, 0].item() == expected_truncated

    # Padded observations repeat the final next observation.
    final_state = batch.observations.vector_state[0, 2]

    for step in range(3, 6):
        torch.testing.assert_close(
            batch.observations.vector_state[0, step],
            final_state,
        )

    # Metadata should contain only real transitions.
    assert batch.train_data == [[{"step": 1}, {"step": 2}]]


def test_sample_sequences_marl_tensors(monkeypatch):
    """Check MARL agent tensors, padding and metadata."""

    buffer = MARLMemoryBuffer(max_capacity=20)

    for i in range(6):
        buffer.add(_make_sequence_marl_experience(i, done=i == 2))

    _force_sequence_start(monkeypatch, buffer, 1)

    batch, indices = memory_sampler.sample_sequences(
        memory=buffer,
        batch_size=1,
        horizon=4,
        device=torch.device("cpu"),
    )

    np.testing.assert_array_equal(indices, [1])

    assert batch.observations.global_state.shape[1] == 5
    assert batch.valid.shape == (1, 4, 1)

    np.testing.assert_array_equal(
        batch.valid[0, :, 0].numpy(),
        [1, 1, 0, 0],
    )

    assert batch.train_data == [[{"step": 1}, {"step": 2}]]

    for agent in batch.actions:
        assert batch.actions[agent].shape == (1, 4, 1)
        assert batch.rewards[agent].shape == (1, 4, 1)

        np.testing.assert_array_equal(
            batch.rewards[agent][0, :, 0].numpy(),
            [2, 3, 0, 0],
        )

        assert batch.terminated[agent][0, 1, 0].item() == 1.0

        assert torch.all(batch.actions[agent][0, 2:] == 0)


def test_sample_sequences_priority(monkeypatch):
    """Prioritized sequence sampling preserves starting indices and weights."""

    buffer = SARLMemoryBuffer(
        max_capacity=20,
        beta=0.0,
        d_beta=0.0,
    )

    for i in range(6):
        buffer.add(_make_sequence_sarl_experience(i))

    sample = Sample(
        experiences=[buffer.memory_buffers[2]],
        indices=[2],
        weights=np.array([0.75]),
    )

    monkeypatch.setattr(
        buffer,
        "sample_priority",
        lambda *args, **kwargs: sample,
    )

    batch, indices = memory_sampler.sample_sequences(
        memory=buffer,
        batch_size=1,
        horizon=3,
        device=torch.device("cpu"),
        use_per_buffer=1,
    )

    np.testing.assert_array_equal(indices, [2])

    assert batch.weights.item() == pytest.approx(0.75)
    assert batch.train_data == [[{"step": 2}, {"step": 3}, {"step": 4}]]


def test_reward_scale_add_overwrite_clear():
    """Reward scaling tracks only experiences currently stored."""

    buffer = SARLMemoryBuffer(max_capacity=3)

    assert buffer.reward_scale() == 1.0

    for i in range(3):
        buffer.add(_make_sequence_sarl_experience(i))

    # Rewards: [1, 2, 3]
    assert buffer.reward_scale() == pytest.approx(2.0)

    # Overwrite reward 1 with reward 10.
    buffer.add(_make_sequence_sarl_experience(9))

    # Rewards: [10, 2, 3]
    assert buffer.reward_scale() == pytest.approx(5.0)

    buffer.clear()

    assert len(buffer) == 0
    assert buffer.reward_scale() == 1.0


def test_reward_scale_n_step():
    """Reward scale uses stored n-step returns, not raw incoming rewards."""

    buffer = SARLMemoryBuffer(
        max_capacity=10,
        n_step=3,
        gamma=1.0,
    )

    buffer.add(_make_sequence_sarl_experience(0))
    buffer.add(_make_sequence_sarl_experience(1))

    assert len(buffer) == 0
    assert buffer.reward_scale() == 1.0

    buffer.add(_make_sequence_sarl_experience(2))

    assert len(buffer) == 1

    expected = abs(buffer.memory_buffers[0].reward)

    assert buffer.reward_scale() == pytest.approx(expected)
