"""
Training utilities for reinforcement learning algorithms.

Common functions used across different RL algorithms to reduce code duplication
while maintaining readability for students.
"""

from typing import Tuple

import numpy as np
import torch

from cares_reinforcement_learning.memory import MemoryBuffer
from cares_reinforcement_learning.util.training_context import (
    Observation,
    ObservationTensors,
)


def consecutive_sample_batch_to_tensors(
    memory: MemoryBuffer,
    batch_size: int,
    device: torch.device,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    # Sample consecutive batch from memory buffer - this returns the full consecutive interface
    # state_i, action_i, reward_i, next_state_i, done_i, ..._i, state_i+1, action_i+1, reward_i+1, next_state_i+1, done_i+1, ..._+i
    experiences = memory.sample_consecutive(batch_size)
    (
        states_t1,
        actions_t1,
        rewards_t1,
        next_states_t1,
        dones_t1,
        states_t2,
        actions_t2,
        rewards_t2,
        next_states_t2,
        dones_t2,
        indices,
    ) = experiences

    batch_size = len(states_t1)

    # Convert to PyTorch tensors with specified dtypes
    (
        states_t1_tensor,
        actions_t1_tensor,
        rewards_t1_tensor,
        next_states_t1_tensor,
        dones_t1_tensor,
        _,
    ) = batch_to_tensors(
        states_t1,
        actions_t1,
        rewards_t1,
        next_states_t1,
        dones_t1,
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
        states_t2_tensor,
        actions_t2_tensor,
        rewards_t2_tensor,
        next_states_t2_tensor,
        dones_t2_tensor,
        _,
    ) = batch_to_tensors(
        states_t2,
        actions_t2,
        rewards_t2,
        next_states_t2,
        dones_t2,
        device,
        states_dtype=states_dtype,
        action_dtype=action_dtype,
        rewards_dtype=rewards_dtype,
        next_states_dtype=next_states_dtype,
        dones_dtype=dones_dtype,
        weights_dtype=weights_dtype,
    )

    return (
        states_t1_tensor,
        actions_t1_tensor,
        rewards_t1_tensor,
        next_states_t1_tensor,
        dones_t1_tensor,
        states_t2_tensor,
        actions_t2_tensor,
        rewards_t2_tensor,
        next_states_t2_tensor,
        dones_t2_tensor,
        indices,
    )


def observation_to_tensors(
    observations: list[Observation],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> ObservationTensors:
    num_obs = len(observations)
    first = observations[0]

    # -------------------------------------------------
    # 1. Vector and image states
    # -------------------------------------------------
    vector_shape = observations[0].vector_state.shape
    vectors = np.empty((num_obs, *vector_shape), dtype=np.float32)

    images = np.empty(0)
    if observations[0].image_state is not None:
        image_shape = observations[0].image_state.shape  # type: ignore[union-attr]
        images = np.empty((num_obs, *image_shape), dtype=np.float32)

    for i in range(num_obs):
        vectors[i] = observations[i].vector_state

        if observations[i].image_state is not None:
            images[i] = observations[i].image_state

    vector_state_tensor = torch.tensor(vectors, device=device, dtype=dtype)

    image_state_tensor = None
    if observations[0].image_state is not None:
        image_state_tensor = torch.tensor(images, device=device, dtype=dtype)

        # Normalise states - image portion
        # This because the states are [0-255] and the predictions are [0-1]
        image_state_tensor /= 255.0

    agent_states_tensor: dict[str, torch.Tensor] | None = None
    avail_actions_tensor = None
    if first.agent_states is not None:
        # -------------------------------------------------
        # 2. Per-agent observations (dict[str → (batch, obs_dim_i)])
        # -------------------------------------------------
        agent_names = list(first.agent_states.keys())  # type: ignore[union-attr]
        agent_states_tensor = {}

        for agent in agent_names:
            # collect obs across batch for a single agent
            obs_list = [s.agent_states[agent] for s in observations]  # type: ignore[index]
            agent_states_tensor[agent] = torch.as_tensor(
                np.stack(obs_list, axis=0),  # (batch, obs_dim_i)
                dtype=dtype,
                device=device,
            )

        # -------------------------------------------------
        # 3. Global avail-actions (rectangular)
        #    shape: (batch, n_agents, action_dim)
        # -------------------------------------------------
        avail_actions = [obs.avail_actions for obs in observations]
        avail_actions_tensor = torch.as_tensor(
            np.stack(avail_actions, axis=0),  # type: ignore[arg-type]
            dtype=torch.float32,
            device=device,
        )

    observation_tensor = ObservationTensors(
        vector_state_tensor=vector_state_tensor,
        image_state_tensor=image_state_tensor,
        agent_states_tensor=agent_states_tensor,
        avail_actions_tensor=avail_actions_tensor,
    )

    return observation_tensor


def sample_to_tensors(
    states: list[Observation],
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: list[Observation],
    dones: np.ndarray,
    device: torch.device,
    weights: np.ndarray | None = None,
    states_dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    next_states_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    ObservationTensors,
    torch.Tensor,
    torch.Tensor,
    ObservationTensors,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert numpy arrays to tensors with consistent dtypes."""
    states_tensor = observation_to_tensors(states, device, states_dtype)

    actions_tensor = torch.tensor(
        np.asarray(actions), dtype=action_dtype, device=device
    )

    rewards_tensor = torch.tensor(
        np.asarray(rewards), dtype=rewards_dtype, device=device
    )

    next_states_tensor = observation_to_tensors(next_states, device, next_states_dtype)

    dones_tensor = torch.tensor(np.asarray(dones), dtype=dones_dtype, device=device)

    if weights is None:
        weights = np.array([1.0] * len(states))

    weights_tensor = torch.tensor(
        np.asarray(weights), dtype=weights_dtype, device=device
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


def sample(
    memory: MemoryBuffer,
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
) -> Tuple[
    ObservationTensors,
    torch.Tensor,
    torch.Tensor,
    ObservationTensors,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:

    # Sample from memory buffer
    if use_per_buffer:
        experiences = memory.sample_priority(
            batch_size,
            sampling_strategy=per_sampling_strategy,
            weight_normalisation=per_weight_normalisation,
        )
        states, actions, rewards, next_states, dones, indices, weights = experiences
    else:
        experiences = memory.sample_uniform(batch_size)
        states, actions, rewards, next_states, dones, indices = experiences
        weights = np.array([1.0] * len(states))

    # Convert to PyTorch tensors with specified dtypes
    (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
    ) = sample_to_tensors(
        states,
        actions,
        rewards,
        next_states,
        dones,
        device,
        weights,
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
        indices,
    )
