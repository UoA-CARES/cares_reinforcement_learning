"""
Memory utilities for reinforcement learning algorithms.
"""

from typing import cast, overload

import numpy as np
import torch

from cares_reinforcement_learning.memory.memory_buffer import MemoryBuffer, Sample
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    MARLObservationTensors,
    SARLObservation,
    SARLObservationTensors,
)


@overload
def observation_to_tensors(
    observations: list[SARLObservation],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SARLObservationTensors: ...


@overload
def observation_to_tensors(
    observations: list[MARLObservation],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> MARLObservationTensors: ...


def observation_to_tensors(
    observations: list[SARLObservation] | list[MARLObservation],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SARLObservationTensors | MARLObservationTensors:
    """Convert observations to tensors, handling both SARL and MARL types."""
    num_obs = len(observations)

    if isinstance(observations[0], MARLObservation):
        observations = cast(list[MARLObservation], observations)

        first = observations[0]
        global_state_list = [obs.global_state for obs in observations]  # type: ignore[attr-defined]
        global_state_tensor = torch.as_tensor(
            np.stack(global_state_list, axis=0),
            dtype=dtype,
            device=device,
        )

        agent_names = list(first.agent_states.keys())
        agent_states_tensor: dict[str, torch.Tensor] = {}

        for agent in agent_names:
            obs_list = [obs.agent_states[agent] for obs in observations]  # type: ignore[index]
            agent_states_tensor[agent] = torch.as_tensor(
                np.stack(obs_list, axis=0),
                dtype=dtype,
                device=device,
            )

        avail_actions = [obs.avail_actions for obs in observations]
        avail_actions_tensor = torch.as_tensor(
            np.stack(avail_actions, axis=0),
            dtype=torch.float32,
            device=device,
        )

        return MARLObservationTensors(
            global_state_tensor=global_state_tensor,
            agent_states_tensor=agent_states_tensor,
            avail_actions_tensor=avail_actions_tensor,
        )
    elif isinstance(observations[0], SARLObservation):
        observations = cast(list[SARLObservation], observations)
        vector_shape = observations[0].vector_state.shape
        vectors = np.empty((num_obs, *vector_shape), dtype=np.float32)

        images = np.empty(0)
        if observations[0].image_state is not None:
            image_shape = observations[0].image_state.shape
            images = np.empty((num_obs, *image_shape), dtype=np.float32)

        for i in range(num_obs):
            vectors[i] = observations[i].vector_state

            if observations[i].image_state is not None:
                images[i] = observations[i].image_state

        vector_state_tensor = torch.tensor(vectors, device=device, dtype=dtype)

        image_state_tensor = None
        if observations[0].image_state is not None:
            image_state_tensor = torch.tensor(images, device=device, dtype=dtype)
            # Normalise states - image portion (range [0-255] -> [0-1])
            image_state_tensor /= 255.0

        return SARLObservationTensors(
            vector_state_tensor=vector_state_tensor,
            image_state_tensor=image_state_tensor,
        )

    raise ValueError(
        f"Unknown observation type: {type(observations[0])} cannot convert to tensors."
    )


@overload
def sample_to_tensors(
    buffer_sample: Sample[SARLObservation],
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    SARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
]: ...


@overload
def sample_to_tensors(
    buffer_sample: Sample[MARLObservation],
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
]: ...


def sample_to_tensors(
    buffer_sample: Sample[SARLObservation] | Sample[MARLObservation],
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert numpy arrays to tensors with consistent dtypes."""
    states_tensor = observation_to_tensors(buffer_sample.states, device, states_dtype)

    actions_tensor = torch.tensor(
        np.asarray(buffer_sample.actions), dtype=action_dtype, device=device
    )

    rewards_tensor = torch.tensor(
        np.asarray(buffer_sample.rewards), dtype=rewards_dtype, device=device
    )

    next_states_tensor = observation_to_tensors(
        buffer_sample.next_states, device, next_states_dtype
    )

    dones_tensor = torch.tensor(
        np.asarray(buffer_sample.dones), dtype=dones_dtype, device=device
    )

    weights_tensor = torch.tensor(
        np.asarray(buffer_sample.weights), dtype=weights_dtype, device=device
    )

    # Reshape to batch_size
    rewards_tensor = rewards_tensor.unsqueeze(-1)
    dones_tensor = dones_tensor.unsqueeze(-1)
    weights_tensor = weights_tensor.unsqueeze(-1)

    return (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
    )


@overload
def consecutive_sample(
    memory: MemoryBuffer[SARLObservation],
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    SARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors,
    torch.Tensor,
    SARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors,
    torch.Tensor,
    np.ndarray,
]: ...


@overload
def consecutive_sample(
    memory: MemoryBuffer[MARLObservation],
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    MARLObservationTensors,
    torch.Tensor,
    MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    MARLObservationTensors,
    torch.Tensor,
    np.ndarray,
]: ...


def consecutive_sample(
    memory: MemoryBuffer[SARLObservation] | MemoryBuffer[MARLObservation],
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    np.ndarray,
]:
    # Sample consecutive batch from memory buffer - this returns the full consecutive interface
    # state_i, action_i, reward_i, next_state_i, done_i, ..._i, state_i+1, action_i+1, reward_i+1, next_state_i+1, done_i+1, ..._+i
    buffer_sample_one, buffer_sample_two = memory.sample_consecutive(batch_size)

    # Convert to PyTorch tensors with specified dtypes
    (
        observations_t1_tensor,
        actions_t1_tensor,
        rewards_t1_tensor,
        next_observations_t1_tensor,
        dones_t1_tensor,
        _,
    ) = sample_to_tensors(
        buffer_sample_one,
        device,
        states_dtype=states_dtype,
        action_dtype=action_dtype,
        rewards_dtype=rewards_dtype,
        next_states_dtype=next_states_dtype,
        dones_dtype=dones_dtype,
        weights_dtype=weights_dtype,
    )

    # Also convert next_actions for temporal algorithms
    (
        observations_t2_tensor,
        actions_t2_tensor,
        rewards_t2_tensor,
        next_observations_t2_tensor,
        dones_t2_tensor,
        _,
    ) = sample_to_tensors(
        buffer_sample_two,
        device,
        states_dtype=states_dtype,
        action_dtype=action_dtype,
        rewards_dtype=rewards_dtype,
        next_states_dtype=next_states_dtype,
        dones_dtype=dones_dtype,
        weights_dtype=weights_dtype,
    )

    return (
        observations_t1_tensor,
        actions_t1_tensor,
        rewards_t1_tensor,
        next_observations_t1_tensor,
        dones_t1_tensor,
        observations_t2_tensor,
        actions_t2_tensor,
        rewards_t2_tensor,
        next_observations_t2_tensor,
        dones_t2_tensor,
        np.asarray(buffer_sample_one.indices),
    )


@overload
def sample(
    memory: MemoryBuffer[SARLObservation],
    batch_size: int,
    device: torch.device,
    use_per_buffer: int = 0,
    per_sampling_strategy: str = "uniform",
    per_weight_normalisation: str = "none",
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    SARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]: ...


@overload
def sample(
    memory: MemoryBuffer[MARLObservation],
    batch_size: int,
    device: torch.device,
    use_per_buffer: int = 0,
    per_sampling_strategy: str = "uniform",
    per_weight_normalisation: str = "none",
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]: ...


def sample(
    memory: MemoryBuffer[SARLObservation] | MemoryBuffer[MARLObservation],
    batch_size: int,
    device: torch.device,
    use_per_buffer: int = 0,
    per_sampling_strategy: str = "uniform",
    per_weight_normalisation: str = "none",
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    SARLObservationTensors | MARLObservationTensors,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:

    # Sample from memory buffer
    if use_per_buffer:
        buffer_sample = memory.sample_priority(
            batch_size,
            sampling_strategy=per_sampling_strategy,
            weight_normalisation=per_weight_normalisation,
        )
    else:
        buffer_sample = memory.sample_uniform(batch_size)

    # Convert to PyTorch tensors with specified dtypes
    (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
    ) = sample_to_tensors(
        buffer_sample=buffer_sample,
        device=device,
        states_dtype=states_dtype,
        action_dtype=action_dtype,
        rewards_dtype=rewards_dtype,
        next_states_dtype=next_states_dtype,
        dones_dtype=dones_dtype,
        weights_dtype=weights_dtype,
    )

    return (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
        np.asarray(buffer_sample.indices),
    )
