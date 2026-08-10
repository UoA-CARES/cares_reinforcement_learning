"""
Memory utilities for reinforcement learning algorithms.
"""

from dataclasses import dataclass
from typing import Any, TypeGuard, cast, overload

import numpy as np
import torch

from cares_reinforcement_learning.memory.memory_buffer import (
    MARLMemoryBuffer,
    Sample,
    SARLMemoryBuffer,
)
from cares_reinforcement_learning.types.experience import (
    MultiAgentExperience,
    SingleAgentExperience,
)
from cares_reinforcement_learning.types.observation import (
    MARLObservation,
    MARLObservationTensors,
    SARLObservation,
    SARLObservationTensors,
)


@dataclass(frozen=True, slots=True)
class SARLTensorSample:
    observation: SARLObservationTensors
    action: torch.Tensor
    reward: torch.Tensor
    next_observation: SARLObservationTensors
    done: torch.Tensor
    weights: torch.Tensor
    train_data: list[dict[str, Any]]


@dataclass(frozen=True, slots=True)
class MARLTensorSample:
    observation: MARLObservationTensors
    action: dict[str, torch.Tensor]
    reward: dict[str, torch.Tensor]
    next_observation: MARLObservationTensors
    done: dict[str, torch.Tensor]
    weights: torch.Tensor
    train_data: list[dict[str, Any]]


def is_sarl_sample(
    buffer_sample: Sample[SingleAgentExperience] | Sample[MultiAgentExperience],
) -> TypeGuard[Sample[SingleAgentExperience]]:
    """Type guard to narrow Sample union to SARL variant."""
    return isinstance(buffer_sample.experiences[0], SingleAgentExperience)


def is_marl_sample(
    buffer_sample: Sample[SingleAgentExperience] | Sample[MultiAgentExperience],
) -> TypeGuard[Sample[MultiAgentExperience]]:
    """Type guard to narrow Sample union to MARL variant."""
    return isinstance(buffer_sample.experiences[0], MultiAgentExperience)


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
        global_state_list = [obs.global_state for obs in observations]
        global_state_tensor = torch.as_tensor(
            np.stack(global_state_list, axis=0),
            dtype=dtype,
            device=device,
        )

        agent_names = list(first.agent_states.keys())
        agent_states_tensor: dict[str, torch.Tensor] = {}
        available_actions_tensor: dict[str, torch.Tensor] = {}

        for agent in agent_names:
            obs_list = [obs.agent_states[agent] for obs in observations]
            agent_states_tensor[agent] = torch.as_tensor(
                np.stack(obs_list, axis=0),
                dtype=dtype,
                device=device,
            )

            available_actions = [obs.available_actions[agent] for obs in observations]
            available_actions_tensor[agent] = torch.as_tensor(
                np.stack(available_actions, axis=0),
                dtype=torch.float32,
                device=device,
            )

        return MARLObservationTensors(
            global_state=global_state_tensor,
            agent_states=agent_states_tensor,
            available_actions=available_actions_tensor,
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
            vector_state=vector_state_tensor,
            image_state=image_state_tensor,
        )

    raise ValueError(
        f"Unknown observation type: {type(observations[0])} cannot convert to tensors."
    )


def _sample_to_tensors_sarl(
    buffer_sample: Sample[SingleAgentExperience],
    device: torch.device,
    states_dtype: torch.dtype,
    action_dtype: torch.dtype,
    rewards_dtype: torch.dtype,
    next_states_dtype: torch.dtype,
    dones_dtype: torch.dtype,
    weights_dtype: torch.dtype,
) -> SARLTensorSample:
    observations, actions, rewards, next_observations, dones, train_data = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for exp in buffer_sample.experiences:
        observations.append(exp.observation)
        actions.append(exp.action)
        rewards.append(exp.reward)
        next_observations.append(exp.next_observation)
        dones.append(exp.done)
        train_data.append(exp.train_data)

    observation_tensor = observation_to_tensors(observations, device, states_dtype)
    actions_tensor = torch.tensor(np.stack(actions), dtype=action_dtype, device=device)
    rewards_tensor = torch.tensor(np.stack(rewards), dtype=rewards_dtype, device=device)
    next_observation_tensor = observation_to_tensors(
        next_observations, device, next_states_dtype
    )
    dones_tensor = torch.tensor(np.stack(dones), dtype=dones_dtype, device=device)
    weights_tensor = torch.tensor(
        buffer_sample.weights, dtype=weights_dtype, device=device
    )

    rewards_tensor = rewards_tensor.unsqueeze(-1)
    dones_tensor = dones_tensor.unsqueeze(-1)
    weights_tensor = weights_tensor.unsqueeze(-1)

    return SARLTensorSample(
        observation=observation_tensor,
        action=actions_tensor,
        reward=rewards_tensor,
        next_observation=next_observation_tensor,
        done=dones_tensor,
        weights=weights_tensor,
        train_data=train_data,
    )


def _sample_to_tensors_marl(
    buffer_sample: Sample[MultiAgentExperience],
    device: torch.device,
    states_dtype: torch.dtype,
    action_dtype: torch.dtype,
    rewards_dtype: torch.dtype,
    next_states_dtype: torch.dtype,
    dones_dtype: torch.dtype,
    weights_dtype: torch.dtype,
) -> MARLTensorSample:
    observations: list[MARLObservation] = []
    next_observations: list[MARLObservation] = []
    train_data: list[dict[str, Any]] = []

    first_exp = buffer_sample.experiences[0]
    agent_ids = list(first_exp.observation.agent_states.keys())

    actions: dict[str, list[np.ndarray]] = {agent_id: [] for agent_id in agent_ids}
    rewards: dict[str, list[Any]] = {agent_id: [] for agent_id in agent_ids}
    dones: dict[str, list[Any]] = {agent_id: [] for agent_id in agent_ids}

    for exp in buffer_sample.experiences:
        observations.append(exp.observation)
        next_observations.append(exp.next_observation)

        for agent_id in agent_ids:
            actions[agent_id].append(exp.action[agent_id])
            rewards[agent_id].append(exp.reward[agent_id])
            dones[agent_id].append(exp.done[agent_id])

        train_data.append(exp.train_data)

    observation_tensors = observation_to_tensors(observations, device, states_dtype)
    next_observation_tensors = observation_to_tensors(
        next_observations, device, next_states_dtype
    )

    action_tensors: dict[str, torch.Tensor] = {}
    reward_tensors: dict[str, torch.Tensor] = {}
    done_tensors: dict[str, torch.Tensor] = {}

    for agent_id in agent_ids:
        action_tensors[agent_id] = torch.as_tensor(
            np.stack(
                [exp.action[agent_id] for exp in buffer_sample.experiences],
                axis=0,
            ),
            dtype=action_dtype,
            device=device,
        )

        reward_tensors[agent_id] = torch.as_tensor(
            np.stack(
                [exp.reward[agent_id] for exp in buffer_sample.experiences],
                axis=0,
            ),
            dtype=rewards_dtype,
            device=device,
        ).unsqueeze(-1)

        done_tensors[agent_id] = torch.as_tensor(
            np.stack(
                [exp.done[agent_id] for exp in buffer_sample.experiences],
                axis=0,
            ),
            dtype=dones_dtype,
            device=device,
        ).unsqueeze(-1)

    weights_tensor = torch.tensor(
        buffer_sample.weights, dtype=weights_dtype, device=device
    )

    weights_tensor = weights_tensor.unsqueeze(-1)

    return MARLTensorSample(
        observation=observation_tensors,
        action=action_tensors,
        reward=reward_tensors,
        next_observation=next_observation_tensors,
        done=done_tensors,
        weights=weights_tensor,
        train_data=train_data,
    )


@overload
def sample_to_tensors(
    buffer_sample: Sample[SingleAgentExperience],
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> SARLTensorSample: ...


@overload
def sample_to_tensors(
    buffer_sample: Sample[MultiAgentExperience],
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> MARLTensorSample: ...


def sample_to_tensors(
    buffer_sample: Sample[SingleAgentExperience] | Sample[MultiAgentExperience],
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> SARLTensorSample | MARLTensorSample:
    if is_sarl_sample(buffer_sample):
        return _sample_to_tensors_sarl(
            buffer_sample,
            device,
            states_dtype,
            action_dtype,
            rewards_dtype,
            next_states_dtype,
            dones_dtype,
            weights_dtype,
        )
    if is_marl_sample(buffer_sample):
        return _sample_to_tensors_marl(
            buffer_sample,
            device,
            states_dtype,
            action_dtype,
            rewards_dtype,
            next_states_dtype,
            dones_dtype,
            weights_dtype,
        )
    raise TypeError(
        "buffer_sample must be Sample[SingleAgentExperience] or Sample[MultiAgentExperience]"
    )


@overload
def consecutive_sample(
    memory: SARLMemoryBuffer,
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    tuple[SARLTensorSample, SARLTensorSample],
    np.ndarray,
]: ...


@overload
def consecutive_sample(
    memory: MARLMemoryBuffer,
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    tuple[MARLTensorSample, MARLTensorSample],
    np.ndarray,
]: ...


def consecutive_sample(
    memory: SARLMemoryBuffer | MARLMemoryBuffer,
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    tuple[SARLTensorSample, SARLTensorSample]
    | tuple[MARLTensorSample, MARLTensorSample],
    np.ndarray,
]:
    # Sample consecutive batch from memory buffer - this returns the full consecutive interface
    # state_i, action_i, reward_i, next_state_i, done_i, ..._i, state_i+1, action_i+1, reward_i+1, next_state_i+1, done_i+1, ..._+i
    buffer_sample_one, buffer_sample_two = memory.sample_consecutive(batch_size)

    # Convert to PyTorch tensors with specified dtypes
    sample_tensors_one = sample_to_tensors(
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
    sample_tensors_two = sample_to_tensors(
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
        (sample_tensors_one, sample_tensors_two),
        np.asarray(buffer_sample_one.indices),
    )  # type: ignore[return-value]


@overload
def sample(
    memory: SARLMemoryBuffer,
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
    SARLTensorSample,
    np.ndarray,
]: ...


@overload
def sample(
    memory: MARLMemoryBuffer,
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
    MARLTensorSample,
    np.ndarray,
]: ...


def sample(
    memory: SARLMemoryBuffer | MARLMemoryBuffer,
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
) -> tuple[SARLTensorSample | MARLTensorSample, np.ndarray]:

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
    sample_tensors = sample_to_tensors(
        buffer_sample=buffer_sample,
        device=device,
        states_dtype=states_dtype,
        action_dtype=action_dtype,
        rewards_dtype=rewards_dtype,
        next_states_dtype=next_states_dtype,
        dones_dtype=dones_dtype,
        weights_dtype=weights_dtype,
    )

    return (sample_tensors, np.asarray(buffer_sample.indices))
