"""
Training utilities for reinforcement learning algorithms.

Common functions used across different RL algorithms to reduce code duplication
while maintaining readability for students.
"""

from typing import Tuple

import numpy as np
import torch

from cares_reinforcement_learning.memory import MemoryBuffer


def batch_to_tensors(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Convert numpy arrays to tensors with consistent dtypes."""
    states_tensor = torch.tensor(np.asarray(states), dtype=states_dtype, device=device)
    actions_tensor = torch.tensor(
        np.asarray(actions), dtype=action_dtype, device=device
    )
    rewards_tensor = torch.tensor(
        np.asarray(rewards), dtype=rewards_dtype, device=device
    )
    next_states_tensor = torch.tensor(
        np.asarray(next_states), dtype=next_states_dtype, device=device
    )
    dones_tensor = torch.tensor(np.asarray(dones), dtype=dones_dtype, device=device)

    if weights is None:
        weights = np.array([1.0] * len(states))

    weights_tensor = torch.tensor(
        np.asarray(weights), dtype=weights_dtype, device=device
    )

    # Reshape to batch_size
    batch_size = len(rewards_tensor)
    rewards_tensor = rewards_tensor.reshape(batch_size, 1)
    dones_tensor = dones_tensor.reshape(batch_size, 1)
    weights_tensor = weights_tensor.reshape(batch_size, 1)

    return (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
    )


def image_state_to_tensors(
    state: dict[str, np.ndarray],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    # For single inference, memory copying is less of a concern than safety
    vector_tensor = torch.tensor(state["vector"], dtype=dtype, device=device)
    vector_tensor = vector_tensor.unsqueeze(0)

    image_tensor = torch.tensor(state["image"], dtype=dtype, device=device)
    image_tensor = image_tensor.unsqueeze(0)

    # Normalise states - image portion
    # This because the states are [0-255] and the predictions are [0-1]
    image_tensor = image_tensor / 255

    return {"image": image_tensor, "vector": vector_tensor}


def image_states_to_tensors(
    states: list[dict[str, np.ndarray]],
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> dict[str, torch.Tensor]:
    states_images = [state["image"] for state in states]
    states_vector = [state["vector"] for state in states]

    # For image data, use from_numpy for memory efficiency (images can be large)
    # Note: This creates a view sharing memory with original numpy array
    # This is acceptable for training as the numpy arrays are typically not modified
    states_images_tensor = torch.from_numpy(np.asarray(states_images)).to(
        dtype=dtype, device=device
    )

    # For vector data, use safer tensor creation (smaller memory footprint)
    states_vector_tensor = torch.tensor(
        np.asarray(states_vector), dtype=dtype, device=device
    )

    # Normalise states and next_states - image portion
    # This because the states are [0-255] and the predictions are [0-1]
    states_images_tensor = states_images_tensor / 255

    return {"image": states_images_tensor, "vector": states_vector_tensor}


def image_batch_to_tensors(
    states: list[dict[str, np.ndarray]],
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: list[dict[str, np.ndarray]],
    dones: np.ndarray,
    device: torch.device,
    weights: np.ndarray | None = None,
    dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
]:
    """Convert multimodal RL batch (states as dicts) to tensors."""
    # Convert multimodal states and next_states
    states_tensors = image_states_to_tensors(states, device, dtype)
    next_states_tensors = image_states_to_tensors(next_states, device, dtype)

    # Convert regular RL components
    actions_tensor = torch.tensor(
        np.asarray(actions), dtype=action_dtype, device=device
    )
    rewards_tensor = torch.tensor(
        np.asarray(rewards), dtype=rewards_dtype, device=device
    )
    dones_tensor = torch.tensor(np.asarray(dones), dtype=dones_dtype, device=device)

    if weights is None:
        weights = np.array([1.0] * len(states))
    weights_tensor = torch.tensor(
        np.asarray(weights), dtype=weights_dtype, device=device
    )

    # Reshape to batch_size
    batch_size = len(states)
    rewards_tensor = rewards_tensor.reshape(batch_size, 1)
    dones_tensor = dones_tensor.reshape(batch_size, 1)
    weights_tensor = weights_tensor.reshape(batch_size, 1)

    return (
        states_tensors,
        actions_tensor,
        rewards_tensor,
        next_states_tensors,
        dones_tensor,
        weights_tensor,
    )


def sample_batch_to_tensors(
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
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    """
    Sample a batch from memory and convert to tensors ready for training.

    This function handles both uniform and priority sampling, converts numpy arrays
    to PyTorch tensors, and reshapes them appropriately for training.

    Args:
        memory: The replay buffer to sample from
        batch_size: Number of experiences to sample
        device: PyTorch device to place tensors on
        use_per_buffer: Whether to use Prioritized Experience Replay
        per_sampling_strategy: Strategy for PER sampling
        per_weight_normalisation: Weight normalization strategy for PER
        states_dtype: Dtype for states tensor (default: torch.float32)
        action_dtype: Dtype for actions tensor (default: torch.float32)
        rewards_dtype: Dtype for rewards tensor (default: torch.float32)
        next_states_dtype: Dtype for next_states tensor (default: torch.float32)
        dones_dtype: Dtype for dones tensor (default: torch.long)
        weights_dtype: Dtype for weights tensor (default: torch.float32)

    Returns:
        Tuple of (states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                 dones_tensor, weights_tensor, indices)
    """

    # Sample from memory buffer
    weights = None
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

    batch_size = len(states)

    # Convert to PyTorch tensors with specified dtypes
    (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
    ) = batch_to_tensors(
        states,
        actions,
        rewards,
        next_states,
        dones,
        device,
        weights=weights,
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


def sample_image_batch_to_tensors(
    memory: MemoryBuffer,
    batch_size: int,
    device: torch.device,
    use_per_buffer: int = 0,
    per_sampling_strategy: str = "uniform",
    per_weight_normalisation: str = "none",
    dtype: torch.dtype = torch.float32,
    action_dtype: torch.dtype = torch.float32,
    rewards_dtype: torch.dtype = torch.float32,
    dones_dtype: torch.dtype = torch.long,
    weights_dtype: torch.dtype = torch.float32,
) -> Tuple[
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    dict[str, torch.Tensor],
    torch.Tensor,
    torch.Tensor,
    np.ndarray,
]:
    """
    Sample a multimodal batch from memory and convert to tensors ready for training.

    This function handles both uniform and priority sampling for multimodal RL data
    where states are dictionaries with 'image' and 'vector' keys.

    Args:
        memory: The replay buffer to sample from
        batch_size: Number of experiences to sample
        device: PyTorch device to place tensors on
        use_per_buffer: Whether to use Prioritized Experience Replay
        per_sampling_strategy: Strategy for PER sampling
        per_weight_normalisation: Weight normalization strategy for PER
        dtype: Dtype for states tensors (default: torch.float32)
        action_dtype: Dtype for actions tensor (default: torch.float32)
        rewards_dtype: Dtype for rewards tensor (default: torch.float32)
        dones_dtype: Dtype for dones tensor (default: torch.long)
        weights_dtype: Dtype for weights tensor (default: torch.float32)

    Returns:
        Tuple of (states_tensor, actions_tensor, rewards_tensor, next_states_tensor,
                 dones_tensor, weights_tensor, indices)
        - states_tensor and next_states_tensor are dict[str, torch.Tensor] with 'image' and 'vector' keys
    """

    # Sample from memory buffer
    weights = None
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

    batch_size = len(states)

    # Convert to PyTorch tensors with specified dtypes for multimodal data
    (
        states_tensor,
        actions_tensor,
        rewards_tensor,
        next_states_tensor,
        dones_tensor,
        weights_tensor,
    ) = image_batch_to_tensors(
        states,
        actions,
        rewards,
        next_states,
        dones,
        device,
        weights=weights,
        dtype=dtype,
        action_dtype=action_dtype,
        rewards_dtype=rewards_dtype,
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
) -> Tuple[
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
    """
    Sample a consecutive batch from memory and convert to tensors ready for training.

    This function handles consecutive sampling, converts numpy arrays to PyTorch tensors,
    and reshapes them appropriately for training. Returns the full consecutive sampling
    interface with both timesteps for temporal algorithms.

    Args:
        memory: The replay buffer to sample from
        batch_size: Number of experiences to sample
        device: PyTorch device to place tensors on
        states_dtype: Dtype for states tensor (default: torch.float32)
        action_dtype: Dtype for actions tensor (default: torch.float32)
        rewards_dtype: Dtype for rewards tensor (default: torch.float32)
        next_states_dtype: Dtype for next_states tensor (default: torch.float32)
        dones_dtype: Dtype for dones tensor (default: torch.long)
        weights_dtype: Dtype for weights tensor (default: torch.float32)

    Returns:
        Tuple of (states_t1_tensor, actions_t1_tensor, rewards_t1_tensor, next_states_t1_tensor,
                 dones_t1_tensor, states_t2_tensor, actions_t2_tensor, rewards_t2_tensor,
                 next_states_t2_tensor, dones_t2_tensor, indices)
    """

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
